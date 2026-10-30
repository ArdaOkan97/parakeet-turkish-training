"""Small shared-head named mapping/step/poison fixture; stdlib and explicit mocks."""
import argparse, ast, hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/parakeet_turkish/bpe_newrows_shared_head_adaptation_v1.py'

def main(args):
 spec=importlib.util.spec_from_file_location('shared_head_fixture',SOURCE)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 checks=[];rejected=[]
 def check(value,label):
  if not value:raise AssertionError(label)
  checks.append(label)
 def reject(call,label):
  try:call()
  except ValueError:rejected.append(label)
  else:raise AssertionError(label)
 shared=m.SHARED_HEAD_NAMES
 heads=(*shared[:2],m.TABLES[0],*shared[2:],*m.TABLES[1:])
 adapters=tuple(f'encoder.layers.{layer}.self_attn.{projection}.lora_{side}' for layer in range(24)
    for projection in ('q_proj','k_proj','v_proj','o_proj') for side in ('A','B'))
 groups=[{'params':list(range(15))},{'params':list(range(15,207))}]
 for names in (heads+adapters,adapters+heads):
  got=m.source_shared_mapping(names,groups)
  check(got[0]==heads and got[1]==adapters and got[2]==shared,'pure named extraction preserves original group order '+str(names[0]))
  check(got[3]==tuple((i,name) for i,name in enumerate(heads) if name not in m.TABLES)
    and set(i for i,_ in got[3])==set(range(15))-{2,13,14},'only original12head IDs copied; resized3/LoRA192 excluded '+str(names[0]))
 for names,bad,label in ((heads+adapters,[{'params':list(reversed(range(15)))},groups[1]],'wrong head IDorder'),
   (heads+adapters[:-1]+(heads[0],),groups,'duplicate source name'),
   (('wrong.shared.weight',)+heads[1:]+adapters,groups,'unknown shared head name')):
  reject(lambda n=names,g=bad:m.source_shared_mapping(n,g),label)
 tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
 methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
 ctor=ast.unparse(methods['__init__']);step=ast.unparse(methods['step']);load=ast.unparse(methods['load_phase_state'])
 check(ctor.index('_compare_initial_weights')<ctor.index('_audit_original_shared')<ctor.index('phase1.load_phase_state')
    <ctor.index('self.row_optimizer.param_groups[0]')<ctor.index('self.shared_optimizer.load_state_dict')
    <ctor.index('self.shared_optimizer.param_groups[0]'), 'raw204/protected audit then source12 moments and phase1load before LR changes')
 check('source[\'optimizer\'][\'param_groups\'][0]' in ctor and 'copy.deepcopy(source[\'optimizer\'][\'state\'][source_id])' in ctor,
       'shared Adam group0 source states copied deeply by name; no full207 load')
 check('self.adapter_names + TABLES' in ast.unparse(methods['_fixed_digest'])
    and 'self.shared_names' not in ast.unparse(methods['_fixed_digest']), 'durable digest protects192LoRA+old/six/blank/duration while shared12 can learn')
 check(step.index('grad[:8199].zero_')<step.index('clip_grad_norm_')<step.index('self._incomplete_update = True')
    <step.index('row_optimizer.step')<step.index('shared_optimizer.step')<step.index('self.phase2_updates = next_count')
    <step.index('self._incomplete_update = False'),'mask once global15clip then poison before BOTHsteps and commit after both')
 check(load.index('_fixed_digest')<load.index('_validate_optimizer_state')<load.index('copy_(value)')
    <load.index('row_optimizer.load_state_dict')<load.index('shared_optimizer.load_state_dict')
    <load.index('self._incomplete_update = False'),'all207/fixed192+rows/bothMom audit before weights then both restore before poison clears')
 check('self.acquisition_updates + count' in load and 'self.source_step + count' in load
    and 'shared_Adam_clock' in load and 'row_Adam_clock' in load,'own restore binds localN vs rowK+N/sharedS+N before mutation')
 class Grad:
  is_sparse=False;shape=(1,)
  def __init__(self,events):self.events=events
  def __getitem__(self,item):self.events.append(('mask_slice',item.start,item.stop));return self
  def zero_(self):self.events.append('zero')
 def make(fail=None):
  x=m.BPENewRowsSharedHeadAdaptation.__new__(m.BPENewRowsSharedHeadAdaptation)
  x.source_step=6500;x.acquisition_updates=500;x.phase2_updates=0;x._incomplete_update=False
  x._check_contract=lambda:None;events=[]
  x.tables=tuple(SimpleNamespace(shape=(1,),grad=Grad(events)) for i in range(3))
  x.shared=tuple(SimpleNamespace(shape=(1,),grad=Grad(events)) for i in range(12))
  clocks={'row':500,'shared':6500}
  def rowstep():
   events.append('row_step');clocks['row']+=1
   if fail=='first':raise RuntimeError('mock first Adam failure')
  def sharedstep():
   events.append('shared_step')
   if fail=='second':raise RuntimeError('mock second Adam failure')
   clocks['shared']+=1
  def clip(parameters,norm,**kw):events.append(('clip',len(parameters),norm,kw['error_if_nonfinite']));return 'norm'
  x.row_optimizer=SimpleNamespace(step=rowstep,zero_grad=lambda **kw:events.append('row_zero'))
  x.shared_optimizer=SimpleNamespace(step=sharedstep,zero_grad=lambda **kw:events.append('shared_zero'))
  x.torch=SimpleNamespace(nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=clip)))
  return x,events,clocks
 x,events,clocks=make();check(x.step()=='norm' and x.phase2_updates==x.optimizer_updates_completed==1
    and x.cumulative_step==7001 and clocks=={'row':501,'shared':6501},'actual pure step API mock commits N1/row501/shared6501/global7001')
 check(events[-3:]==[('clip',15,1.,True),'row_step','shared_step']
    and events.count('zero')==6,'only3table masks before one global15 norm and two steps')
 for failure in ('first','second'):
  x,events,clocks=make(failure)
  try:x.step()
  except RuntimeError:pass
  else:raise AssertionError('failed Adam accepted')
  check(x._incomplete_update and x.phase2_updates==0,'failed '+failure+' Adam leaves poisoned boundary/no local commit')
  reject(lambda:x.zero_grad(),'poisoned '+failure+' rejects zero')
  reject(lambda:x.state_dict(),'poisoned '+failure+' rejects durable save')
 check(all(name in methods for name in ('phase_identity','state_dict','load_phase_state','load_state_dict','unfreeze'))
    and m.KIND==m.STATE_KIND=='HF_TSC_BPE512_new_rows_plus_shared_heads_state_v1','distinct full207/twoAdam ownserializer and restore APIs')
 check('source_shared_weight_decay_retained' in ast.unparse(methods['phase_identity'])
    and "self.shared_group['weight_decay'] == 0.01" in ast.unparse(methods['_check_contract']), 'original shared decay.01 retained; row decay0')
 check('torch' not in sys.modules and 'nemo' not in sys.modules,'no ML/Torch/tensors/checkpoints/models/GPU/jobs')
 output=args.output.resolve();assert not output.exists(),'Fresh fixture receipt required'
 report={'schema':1,'kind':'BPE_shared_head_adaptation_stdlib_verification_v1','passed':True,
  'utility_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'checks':checks,'checks_passed':len(checks),'rejections':rejected,'rejections_passed':len(rejected),
  'scope':'Real pure name mapping and step/poison APIs with explicit gradients/optimizers mocks plus AST source ordering. No real 207 tensors/Adam copies or hardware save/load. Actual source300/gradients/deltas/byte equality/native fit/runtime admission remain ROOT gates.'}
 with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejected),'utility_sha256':report['utility_sha256']}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
