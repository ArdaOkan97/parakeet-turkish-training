"""Stdlib-only LR/moment-copy fixture; no tensor/checkpoint/model/GPU imports."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'outputs/parakeet_turkish'
ORIGINAL = HERE / 'bpe_newrows_lora_adaptation_v1.py'
UTILITY = HERE / 'bpe_newrows_lora_adaptation_low_lr_v1.py'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    trees = [ast.parse(path.read_text(encoding='utf-8')) for path in (ORIGINAL, UTILITY)]
    methods = [{n.name: n for n in next(c for c in tree.body if isinstance(c, ast.ClassDef)).body
        if isinstance(n, ast.FunctionDef)} for tree in trees]
    old, new = methods; checks = []
    def check(ok, name):
        assert ok, name
        checks.append(name)
    check(sha(ORIGINAL) == '2f70811d67af5fc950fa547727dc5ae18fa895a1e5e2cb572c4ff40bbc42476b',
        'reviewed original2f708 unchanged')
    for name in ('_compare_initial_weights', '_audit_original_lora', '_check_moments', '_fixed_digest',
                 'step', 'zero_grad', '_require_complete_update', 'state_dict', 'load_phase_state',
                 '_validate_optimizer_state', 'load_state_dict', 'unfreeze'):
        check(ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False),
            'unchanged actual ' + name)
    ctor = ast.unparse(new['__init__'])
    check(ctor.index('self._audit_original_lora') < ctor.index('copy.deepcopy(source[\'optimizer\'][\'state\']')
        < ctor.index('self.lora_optimizer.load_state_dict(target)')
        < ctor.index("self.lora_optimizer.param_groups[0]['lr'] = 3e-05")
        < ctor.index('self.lora_group = copy.deepcopy'), 'source audit/copy/load before ONLY active10xLR reduction')
    statements = new['__init__'].body
    start = next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id == 'original_group' for t in n.targets))
    end = next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Attribute) and t.attr == 'lora_group' for t in n.targets)) + 1
    source_group = {'params': list(range(15,207)), 'lr': 3e-4, 'weight_decay': 0.,
        'betas': (.9,.999), 'eps': 1e-8, 'amsgrad': False, 'maximize': False,
        'capturable': False, 'differentiable': False, 'foreach': None, 'fused': None}
    source_state = {i: {'step':6500, 'exp_avg':[i/100.,-.1], 'exp_avg_sq':[i/200.,.1]}
        for i in range(207)}
    source = {'optimizer': {'param_groups':[{'params':list(range(15))},source_group], 'state':source_state}}
    untouched = copy.deepcopy(source); loads = []
    class MockAdam:
        def __init__(self, groups):
            self.param_groups = groups; self.state = {}
        def state_dict(self):
            return {'param_groups':[dict(self.param_groups[0], params=list(range(192)))], 'state':{}}
        def load_state_dict(self, value):
            loads.append(copy.deepcopy(value))
            assert value['param_groups'][0]['lr'] == 3e-4
            self.state = copy.deepcopy(value['state'])
            group = {k:v for k,v in value['param_groups'][0].items() if k!='params'}
            self.param_groups[0].update(group)
    stage = SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(AdamW=MockAdam)),
        adapters=tuple('LoRA_' + str(i) for i in range(192)),
        row_optimizer=SimpleNamespace(lr=1e-4,clock=500))
    env = {'self':stage,'source':source,'mapping':tuple((i,'LoRA_'+str(i-15)) for i in range(15,207)),
           'copy':copy}
    block = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(statements[start:end]),type_ignores=[]))
    exec(compile(block,str(UTILITY),'exec'),env)
    check(len(loads)==1 and loads[0]['param_groups'][0]['lr']==3e-4 and stage.lora_group['lr']==3e-5,
        'actual constructor-block mock loads source3e-4 then active3e-5')
    check(stage.lora_group['lr'] / source_group['lr'] == .1 and
        {k:v for k,v in stage.lora_group.items() if k!='lr'} ==
        {k:v for k,v in source_group.items() if k not in ('params','lr')}, 'onlyLR changed by factor0.1')
    check(source == untouched and stage.row_optimizer.lr==1e-4 and stage.row_optimizer.clock==500,
        'source207 groups/moments and rowAdamK500 unaffected')
    check(len(stage.lora_optimizer.state)==192 and all(
        stage.lora_optimizer.state[i]==source_state[i+15] for i in range(192)),
        'actual mapped moment values/sourceS6500 clock copied for all192')
    stage.lora_optimizer.state[0]['exp_avg'][0] = 999
    check(source == untouched and env['target']['state'][0]['exp_avg'][0] == source_state[15]['exp_avg'][0],
        'no mutable moment alias back to source207 or stagedcopy')
    constants = {n.targets[0].id:n.value.value for n in trees[1].body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    check(constants['KIND']=='HF_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_state_v1' and
        constants['IDENTITY_KIND']=='HF_TSC_BPE512_new_rows_plus_encoder_LoRA_low_lr_identity_v1',
        'separate declared own-phase state/identity kinds')
    identity = ast.unparse(new['phase_identity'])
    check("'source_original_LoRA_lr': 0.0003" in identity and "'active_LoRA_lr': 3e-05" in identity
        and "'source_LoRA_moments_reset': False" in identity, 'explicit original vsactive LR provenance/noreset')
    check('torch' not in __import__('sys').modules, 'no Torch/ML import')
    report = {'schema':1,'kind':'BPE_newrows_LoRA_low_lr_stdlib_verification_v1','passed':True,
        'utility_sha256':sha(UTILITY),'original_utility_sha256':sha(ORIGINAL),'fixture_sha256':sha(__file__),
        'checks':checks,'checks_passed':len(checks),'actual_CPU_tensor_checks_executed':False,
        'actual_GPU_or_ASR_executed':False,
        'scope':'Actual constructor LR/moment-copy block executed with explicit stdlib Adam/list mocks; raw tensor/moment audit, ASR gradients, quality and hardware fit unmeasured. All original strict guards/serializer/poison/step ASTs unchanged.'}
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'report':str(args.output),'sha256':sha(args.output),
        'utility_sha256':sha(UTILITY),'checks_passed':len(checks)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
