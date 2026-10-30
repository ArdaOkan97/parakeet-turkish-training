"""Source/CLI/state-protocol checks only; no factory admission or Torch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import tsc_uniform_TR_loss_transition_v1 as h
import run_TSC_uniform_TR_loss_v1 as c
import run_TSC_uniform_TR_loss_journal_v1 as a
import record_TSC_uniform_TR_loss_source_v1 as r
import TR_tdt_population_weighting_v1 as loss
checks=[]
def check(ok,name):
    if not ok:raise AssertionError(name)
    checks.append(name)
def function(source,name):
    return next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
old=(OUT/'train_lora_batched_tsc_uniform_head_lr_v1.py').read_text(encoding='utf-8')
new=(OUT/'train_lora_batched_tsc_uniform_TR_loss_v1.py').read_text(encoding='utf-8')
for name in ('save_training_state','validate_own_state','encoder_lr_for_update','evaluate','pt_focused_batch_objective'):
    check(ast.dump(function(old,name),include_attributes=False)==ast.dump(function(new,name),include_attributes=False),'unchanged '+name)
def rng_calls(source):
    return [ast.dump(x,include_attributes=False) for x in ast.walk(function(source,'run')) if isinstance(x,ast.Call) and ast.unparse(x.func) in ('random.seed','random.getstate','random.setstate','torch.manual_seed','torch.get_rng_state','torch.set_rng_state','torch.cuda.manual_seed_all','torch.cuda.get_rng_state_all','torch.cuda.set_rng_state_all')]
check(rng_calls(old)==rng_calls(new),'source and exact Python/CPU/allCUDA RNG calls unchanged')
check(c.caller_line(OUT/c.WORKER)==c.EXPECTED_SYNC_LINE==798,'one actual CUDA-sync dispatch line798')
check(c.source_pins()[loss.__name__+'.py']==hashlib.sha256((OUT/(loss.__name__+'.py')).read_bytes()).hexdigest(),'utility hash in native source pins')
check(a.EXPECTED_CORE==hashlib.sha256((OUT/'run_TSC_uniform_TR_loss_v1.py').read_bytes()).hexdigest(),'journal pins new scientific core')
check(a.EXPECTED_WORKER==hashlib.sha256((OUT/c.WORKER).read_bytes()).hexdigest(),'journal pins new worker')
for mode in loss.MODES:
    args=c.parser().parse_args(['--tr-tdt-weighting',mode,'--head-lr','0.00001','--encoder-lr-schedule','cosine'])
    check(args.tr_tdt_weighting==mode,'controller parser '+mode)
    args=r.parser().parse_args(['--source-review','work/future_not_admitted.json','--target-run-id','future_fixture_only','--encoder-lr-schedule','cosine','--head-lr','0.00001','--tr-tdt-weighting',mode])
    check(args.tr_tdt_weighting==mode,'creator parser '+mode)
# Extract the actual pure parser; avoid the family's top-level Torch imports.
ns={'argparse':argparse,'Path':Path,'DATA':ROOT/'work/data','OUTPUT':OUT,'MODES':loss.MODES,'POPULATION_MEAN_U':loss.POPULATION_MEAN_U}
exec(compile(ast.Module(body=[function(new,'parser')],type_ignores=[]),'worker_parser_AST','exec'),ns)
common=['--tsc-input-proof','work/future.json','--tsc-training-approval','work/future.json','--source-closure','work/future.json','--run-id','future_fixture_only','--replay-seed','7','--resume-uniform-full','work/future.pt','--resume-majority','work/future.pt','--majority-source-receipt','work/future.json','--uniform-full-source-receipt','work/future.json','--uniform-full-runtime-proof','work/future_ready.json','--indexed-tr-directory','work/future_index','--encoder-lr-schedule','cosine']
for mode in loss.MODES:
    args=ns['parser']().parse_args(common+['--tr-tdt-weighting',mode])
    check(args.tr_tdt_weighting==mode and args.tr_tdt_population_mean_U==loss.POPULATION_MEAN_U,'actual worker parser AST '+mode)
check('Completed actual head policy required' in (OUT/'tsc_uniform_TR_loss_transition_v1.py').read_text(encoding='utf-8'),'actual choice gate before admission exists')
check('before checkpoint byte reads' in (OUT/'record_TSC_uniform_TR_loss_source_v1.py').read_text(encoding='utf-8'),'actual choice gate before source copy exists')
check('torch' not in sys.modules,'no Torch import')
paths=('tsc_uniform_TR_loss_transition_v1.py','record_TSC_uniform_TR_loss_source_v1.py','train_lora_batched_tsc_uniform_TR_loss_v1.py','run_TSC_uniform_TR_loss_v1.py','run_TSC_uniform_TR_loss_journal_v1.py','TR_tdt_population_weighting_v1.py')
report={'kind':'inactive_TR_loss_family_focused_source_parser_migration_CPU_verification_v1','passed':True,'checks':checks,'checks_count':len(checks),'source_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in paths},'opaque_state_math_fixture':'TR_tdt_population_weighting_stdlib_CPU_verification_v1.json','actual_source_factory_or_preflight_or_head_policy_admission':False,'checkpoint_or_history_arrays_loaded':False,'Torch_imported':False,'actual_GPU_tested':False,'scope':'AST-preserved actual full207 serializer/own-state/three RNG families, actual pure parsers/sourcepin dispatch; no real tensor/state migration or future head choice claimed.'}
p=OUT/'TR_loss_family_focused_source_parser_CPU_verification_v1.json';assert not p.exists()
p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks':len(checks),'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
