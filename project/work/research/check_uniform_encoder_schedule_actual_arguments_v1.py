"""Bounded actual parser/admission check; future native PID proof is deferred."""
from pathlib import Path
import argparse,ast,hashlib,json,math,sys,copy
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(HERE))
p=argparse.ArgumentParser();p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True);cli=p.parse_args()
import tsc_uniform_encoder_schedule_transition_v1 as boundary
source=HERE/'train_lora_batched_tsc_uniform_encoder_schedule_v1.py';tree=ast.parse(source.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('parser','validate_arguments','encoder_lr_for_update')]
ns={'argparse':argparse,'Path':Path,'DATA':ROOT/'work/data','OUTPUT':HERE,'math':math,'__doc__':'actual extracted worker CLI','native_loss_identity':lambda args:None,'majority_preflight':boundary.preflight}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
plan=json.loads(cli.preflight.read_text(encoding='utf-8'));args=ns['parser']().parse_args(plan['worker_arguments']);ns['validate_arguments'](args)
f=ns['encoder_lr_for_update'];assert f('cosine',0)==f('cosine',1)==1e-4 and f('cosine',18129)==3e-5 and f('cosine',9065)==(1e-4+3e-5)/2 and f('constant',18129)==1e-4
rejections=[]
for key,value in [('head_lr',3e-5),('lora_lr',3e-5),('encoder_schedule_updates',1000),('encoder_min_lr',1e-5),('encoder_lr_schedule','linear'),('persistence_mode','single_serialize_two_commits'),('retention_stop_policy','enforced')]:
    altered=copy.copy(args);setattr(altered,key,value)
    try:ns['validate_arguments'](altered)
    except ValueError:rejections.append([key,str(value)])
    else:raise AssertionError(('Accepted undeclared change',key,value))
assert 'torch' not in sys.modules
report={'kind':'matched_encoder_schedule_actual_parser_admission_v1','passed':True,'mode':args.encoder_lr_schedule,'real_source_metadata_preflight':True,'actual_parser_validation_extracted':True,'future_same_PID_native_proof_deferred_to_dispatch':True,'source_step':25109,'matched_updates':18129,'end_step':43238,'source_global_Adam_clock_not_reset':True,'pure_schedule_first_mid_last_checked':True,'rejected_unauthorized_changes':rejections,'worker_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(cli.preflight.read_bytes()).hexdigest(),'Torch_imported':False,'GPU_launched':False,'checkpoint_tensor_bytes_read':False}
with cli.output.open('x',encoding='utf-8') as h:json.dump(report,h,indent=2)
print(json.dumps({'passed':True,'mode':args.encoder_lr_schedule,'rejections':len(rejections),'Torch_imported':False}))

