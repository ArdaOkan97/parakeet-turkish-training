"""Minimal stdlib source/clock fixture; all optimizer objects are explicit mocks."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
UTILITY=ROOT/'outputs/parakeet_turkish/bpe_newrows_continued_acquisition_v1.py'


def main(args):
    spec=importlib.util.spec_from_file_location('continued_rows_mock',UTILITY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse(UTILITY.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    checks=[]
    def check(ok,name):
        assert ok,name
        checks.append(name)
    constructor=ast.unparse(methods['__init__'])
    check(constructor.index('_compare_source_weights') < constructor.index('BPEOnlyAcquisition(model')
        < constructor.index('lr=0.0001') < constructor.index('load_phase_state(phase1_state)')
        < constructor.index("self.optimizer.param_groups[0]['lr'] = self.active_row_lr"),
        'original204/protected bytes and strictphase1 sourceLR1e4/load before liveLR change')
    check('original207_state[\'optimizer\']' not in constructor and 'phase1_state' in constructor,
        'original207 Adam never loaded into active3 optimizer')
    source=(ROOT/'outputs/parakeet_turkish/bpe_newrows_acquisition_v1.py').read_bytes()
    check(hashlib.sha256(source).hexdigest()==module.SOURCE_UTILITY_SHA, 'strictsource utility pin unchanged')
    step=ast.unparse(methods['step'])
    check(step.index('mask_then_clip') < step.index('self.optimizer.step()') < step.index('self._local_updates += 1'),
        'mask before one3table clip/step/local count commit')
    for lr in (1e-3,3e-4):
        stage=module.BPEContinuedAcquisition.__new__(module.BPEContinuedAcquisition)
        stage.source_step=6500;stage.source_acquisition_updates=500;stage._local_updates=0
        stage._incomplete_update=False;stage.active_row_lr=lr
        tables=tuple(object() for _ in range(3));stage.tables=tables;calls=[]
        optimizer=SimpleNamespace(param_groups=[{'lr':lr}],state={p:{} for p in tables},
            step=lambda:calls.append('step'))
        stage.optimizer=optimizer
        stage._base=SimpleNamespace(optimizer_updates_completed=500,_check_contract=lambda:None,
            mask_then_clip=lambda norm:calls.append(('masked3clip',norm)) or 'norm',zero_grad=lambda:None)
        check(stage.optimizer_updates_completed==0 and stage.cumulative_step==7000,'initial local0/Adam500/global7000 '+str(lr))
        check(stage.step()=='norm' and stage.optimizer_updates_completed==1
            and stage._base.optimizer_updates_completed==501 and stage.cumulative_step==7001
            and calls==[('masked3clip',1.),'step'], 'actual step API mock carries K+N not N '+str(lr))
        def fail(): raise RuntimeError('mocked partial Adam exception')
        optimizer.step=fail
        try:stage.step()
        except RuntimeError:pass
        else:raise AssertionError('failed step accepted')
        check(stage._incomplete_update and stage.optimizer_updates_completed==1, 'failed step poisoned, localcount not advanced '+str(lr))
        try:stage.state_dict()
        except ValueError:pass
        else:raise AssertionError('partial state saved')
    load=ast.unparse(methods['load_phase_state'])
    check(load.index('_fixed_digest') < load.index('_audit_optimizer') < load.index('copy_(value)')
        < load.index('optimizer.load_state_dict') < load.index('self._incomplete_update = False'),
        'ownkind/frozen207/three moment+K+N audit before any copy and poison clears after full restore')
    check(module.STATE_KIND=='HF_TSC_BPE512_new_rows_continued_acquisition_state_v1'
        and all(n in methods for n in ('state_dict','load_phase_state','load_state_dict','unfreeze')),
        'distinct own207/3Adam serializer/load API')
    check('torch' not in sys.modules, 'no ML/tensor/GPU imports')
    report={'schema':1,'kind':'continued_BPE_acquisition_stdlib_verification_v1','passed':True,
        'utility_sha256':hashlib.sha256(UTILITY.read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks,'checks_passed':len(checks),
        'scope':'Actual pure class step API with explicit mocks plus source ordering/schema guards; no real tensor/checkpoint/model/ASR/CPU-Torch/GPU. Source500 moment retention/rawbytes and hardware remain actual ROOT runtime gates.'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'report':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'utility_sha256':report['utility_sha256'],'checks':len(checks)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
