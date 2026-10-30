"""Stdlib-only argv/real-parser checks. No worker/model/Torch imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'outputs/parakeet_turkish'
DATA = ROOT/'work/data'
checks = []


def check(name, value):
    assert value, name
    checks.append(name)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


keys = ('head_lr','lora_lr','new_token_lr_multiplier','replay_kl','duration_kl',
        'feature_distillation','bucket_size','seed','max_tokens','chunk_size',
        'no_encoder_checkpoint','phonetic_sh_donors')
v8 = OUTPUT/'run_experiment_loop_continuation_v8.py'
v9 = OUTPUT/'run_experiment_loop_continuation_v9.py'
trainer = OUTPUT/'train_lora_batched_data_v9.py'
namespace = {'IDENTITY_KEYS':keys, 'require':require, 'scoped':lambda p:Path(p).resolve()}
old = function(v8, 'training_argv', dict(namespace))
new = function(v9, 'training_argv', dict(namespace))
trainer_parser = function(trainer, 'parser', {'argparse':argparse, 'Path':Path,
    'DATA':DATA, 'OUTPUT':OUTPUT, '__doc__':'Real frozen trainer9 parser'})
config = json.loads((OUTPUT/'runs/adaptive_20261011_native_memory_fix_01_native_resume/run_config.json').read_text())
args = SimpleNamespace(screen_updates=250, strong_threshold=.1193939393939394,
    tr_manifest=DATA/'tr_public_v3_train.jsonl', replay_manifest=DATA/'replay_train.jsonl',
    data_proof=OUTPUT/'tr_public_v3_preparation.json',
    tr_frame_metadata=DATA/'tr_public_v3_frame_metadata_v1.jsonl',
    tr_frame_metadata_proof=OUTPUT/'TR_frame_metadata_cpu_verification_v1.json',
    reuse_input_audit=OUTPUT/'public_data_full_audit_reuse_v1/reuse_receipt.json',
    runtime_migration=OUTPUT/'runs/adaptive_20261011_native_memory_fix_01/runtime_migration_cap9.json')
positional=(args,config,'future_cadence_fixture',ROOT/'work/fixture_start.pt',8,30000,10000,ROOT/'work/fixture_stop.txt')
common={'native_ready':OUTPUT/'fixture_native_ready.json'}
for fraction in (.5,.25):
    a=old(*positional,replay_fraction=fraction,**common)
    b=new(*positional,replay_fraction=fraction,**common)
    check('short_'+str(fraction)+'_argv_exact_old',a==b)
    long=new(*positional,replay_fraction=fraction,long_stage=True,**common)
    parsed=trainer_parser().parse_args(long)
    check('long_'+str(fraction)+'_real_trainer_parser',parsed.eval_every_updates==1000
          and parsed.eval_epoch_fraction==250*(16-round(16*fraction))/74883)
    original=trainer_parser().parse_args(a)
    changed={k for k,v in vars(parsed).items() if v!=vars(original)[k]}
    check('long_'+str(fraction)+'_only_cadence_changed',changed=={'eval_every_updates'})
    check('long_'+str(fraction)+'_recipe_native_KD_retained',parsed.tdt_implementation=='nemo'
          and parsed.replay_kl==config['replay_kl'] and parsed.duration_kl==config['duration_kl']
          and parsed.feature_distillation==config['feature_distillation']
          and parsed.native_max_joint_bytes==1700000000)
a=old(*positional,exact_resume=True,**common)
b=new(*positional,exact_resume=True,**common)
check('short_exact_resume_argv_exact_old',a==b)
parsed=trainer_parser().parse_args(b)
check('short_exact_resume_no_fresh_weights',parsed.resume_exact is not None
      and parsed.resume_weights is None and parsed.resume_runtime_migration is not None)
try:
    new(*positional,exact_resume=True,long_stage=True,**common)
except RuntimeError:
    check('wrong_saved_long_cadence_rejected_before_launch',True)
else:
    raise AssertionError('Changing cadence during exact resume must fail')
matching=dict(config,eval_every_updates=1000,eval_epoch_fraction=250*8/74883)
long_positional=(args,matching,*positional[2:])
parsed=trainer_parser().parse_args(new(*long_positional,exact_resume=True,long_stage=True,**common))
check('matching_long_exact_resume_parser',parsed.resume_exact is not None and parsed.eval_every_updates==1000)
for path in (v8,v9):
    tree=ast.parse(path.read_text())
    frozen=next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FROZEN' for t in n.targets))
    if path==v8:old_frozen=frozen
    else:check('all_worker_loss_data_dispatch_pins_unchanged',frozen==old_frozen)
report={'schema':1,'kind':'long_stage_cadence_stdlib_dispatch_verification_v1','passed':True,
    'checks':checks,'checks_passed':len(checks),'Torch_or_NeMo_imported':False,'GPU_used':False,
    'controller_launched':False,'trainer_changed':False,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (v8,v9,trainer)},
    'scope':'Generated argv parsed by actual frozen trainer9 parser AST; no native backend/model/optimizer execution',
    'future_launch_prerequisites':'ROOT must supply fresh namespace and matching reviewed fork/migration inputs; no future recipe selected by these checks.'}
target=OUTPUT/'long_stage_cadence_cpu_verification_v1.json'
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'checks_passed':len(checks),'report':str(target)}))
