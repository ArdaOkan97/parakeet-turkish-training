"""Focused scheduled-row stdlib/API checks; no Torch, source tensors or jobs."""
import argparse, ast, copy, hashlib, importlib.util, json, math, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'outputs/parakeet_turkish'
UTILITY=HERE/'bpe_newrows_scheduled_acquisition_v1.py'

def main(args):
 sys.path.insert(0,str(HERE))
 spec=importlib.util.spec_from_file_location('scheduled_rows_fixture',UTILITY)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from bpe_newrows_continued_acquisition_v1 import BPEContinuedAcquisition as Parent, STATE_KIND as OLD_KIND
 checks=[];rejections=[]
 def check(ok,label):
  if not ok:raise AssertionError(label)
  checks.append(label)
 def reject(call,label):
  try:call()
  except ValueError:rejections.append(label)
  else:raise AssertionError(label)
 def stage(peak):
  x=m.BPEScheduledAcquisition.__new__(m.BPEScheduledAcquisition)
  x._schedule_initialized=True;x.peak_row_lr=x.active_row_lr=peak
  x.source_step=6500;x.source_acquisition_updates=500;x._local_updates=0;x._incomplete_update=False
  x._static_group={'weight_decay':0.,'betas':(.9,.999),'eps':1e-8,'amsgrad':False,'maximize':False}
  x.tables=tuple(object() for _ in range(3));x.last_applied_lr=None;x.calls=[]
  x.optimizer=SimpleNamespace(param_groups=[{**copy.deepcopy(x._static_group),'lr':1e-4}],state={p:{} for p in x.tables})
  x._base=SimpleNamespace(base_lr=peak,optimizer_updates_completed=500,
   declared_optimizer_group={**copy.deepcopy(x._static_group),'lr':1e-4},_check_contract=lambda:None,
   mask_then_clip=lambda norm:x.calls.append(('masked3clip',norm)) or 'norm',zero_grad=lambda:None)
  x.optimizer.step=lambda:x.calls.append(('Adam',x.current_row_lr))
  x._base.phase_identity=lambda:{'kind':'base_mock','base_lr':peak,
   'declared_optimizer_group':copy.deepcopy(x._base.declared_optimizer_group)}
  x.original_record={'path':str(ROOT/'work/opaque_original.pt'),'sha256':'a'*64,'bytes':1}
  x.phase1_record={'path':str(ROOT/'work/opaque_phase1.pt'),'sha256':'b'*64,'bytes':1}
  x.source_phase_identity={'kind':'opaque_source_1e4'};x.fixed_weight_sha256='c'*64
  return x
 tree=ast.parse(UTILITY.read_text(encoding='utf-8-sig'))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)); methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
 constructor=ast.unparse(methods['__init__']); step=ast.unparse(methods['step'])
 check(constructor.index('super().__init__')<constructor.index('_set_dynamic_lr(SOURCE_ROW_LR)'),
       'frozen strictsource constructor/load precedes scheduled group setup')
 old=ast.parse((HERE/'bpe_newrows_continued_acquisition_v1.py').read_text())
 old_init=ast.unparse(next(n for c in old.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
 check(old_init.index('load_phase_state(phase1_state)')<old_init.index("self.optimizer.param_groups[0]['lr'] = self.active_row_lr"),
       'actual retained phase1 group1e4/K audit-load precedes every live LR change')
 check(hashlib.sha256((HERE/'bpe_newrows_continued_acquisition_v1.py').read_bytes()).hexdigest()==m.CONTINUED_UTILITY_SHA,
       'frozen source loader207/3Adam implementation matches pin')
 check(step.index('mask_then_clip')<step.index('_set_dynamic_lr(applied)')<step.index('self.optimizer.step()')<step.index('self._local_updates += 1'),
       'mask then single3tableclip then scheduledLR then Adam then localcommit')
 for peak in m.ALLOWED_PEAKS:
  rates=[m.lr_for_update(i,peak) for i in range(1,1001)]
  check(rates[0]==peak/50 and rates[49]==peak and rates[-1]==peak*.1,'warmup1/50 and final0.1peak endpoints '+str(peak))
  check(all(a<b for a,b in zip(rates[:49],rates[1:50])) and all(a>b for a,b in zip(rates[49:-1],rates[50:])),
        '50point linear warmup followed by strictly declining cosine '+str(peak))
  check(all(math.isfinite(r) and peak*.02<=r<=peak for r in rates),'all1000 schedule values finite/positive/bounded '+str(peak))
  x=stage(peak);before=x.phase_identity();x._check_contract()
  check(x.current_row_lr==1e-4 and x.last_applied_lr is None and x.cumulative_step==7000 and x.optimizer_updates_completed==0,
        'sourcegroup1e4/K500 with localN0 before first scheduled update '+str(peak))
  for i,lr in enumerate(rates,1):
   check_step=x.step();assert check_step=='norm' and x.calls[-2:]==[('masked3clip',1.),('Adam',lr)]
   assert x._local_updates==i and x._base.optimizer_updates_completed==500+i and x.cumulative_step==7000+i
  check(x.optimizer_updates_completed==1000 and x._base.optimizer_updates_completed==1500,
        'actual class step API mock keeps local1000 distinct from Adam1500 '+str(peak))
  check(x.phase_identity()==before and 'declared_optimizer_group' not in before,
        'static phase identity remains identical across every appliedLR '+str(peak))
  reject(lambda:x.step(),'update1001 rejected '+str(peak))
  x=stage(peak)
  x.optimizer.step=lambda:(_ for _ in ()).throw(RuntimeError('mock Adam exception'))
  try:x.step()
  except RuntimeError:pass
  else:raise AssertionError('failed Adam accepted')
  check(x._incomplete_update and x.optimizer_updates_completed==0 and x.current_row_lr==rates[0],
        'failed Adam poisons save/zero while localcount unadvanced '+str(peak))
  reject(lambda:x.zero_grad(),'poison rejects zero '+str(peak));reject(lambda:x.state_dict(),'poison rejects durable save '+str(peak))
 for update in (0,1001,True,1.5):reject(lambda u=update:m.lr_for_update(u,1e-3),'invalid schedule update '+repr(update))
 reject(lambda:m.lr_for_update(1,2e-3),'undeclared peak rejects')
 # Own restore dispatch is real; only inherited tensor loader is explicitly
 # mocked. It must receive a translated known kind after schedule validation.
 old_load=Parent.load_phase_state;old_save=Parent.state_dict;old_audit=Parent._audit_optimizer
 try:
  received=[]
  def loader(x,state):
   received.append(copy.deepcopy(state));assert state['kind']==OLD_KIND
   x._incomplete_update=True;x._local_updates=state['continued_acquisition_updates']
   x._base.optimizer_updates_completed=state['acquisition_optimizer_updates']
   x.optimizer.param_groups[0]['lr']=state['current_row_lr'];x._check_contract();x._incomplete_update=False
   return copy.deepcopy(state['runtime_state'])
  Parent.load_phase_state=loader
  x=stage(3e-4);n=75
  saved={'schema':1,'kind':m.STATE_KIND,'phase_identity':x.phase_identity(),'row_schedule':m.schedule_identity(3e-4),
   'scheduled_acquisition_updates':n,'continued_acquisition_updates':n,'acquisition_optimizer_updates':500+n,
   'cumulative_step':7000+n,'last_applied_lr':x.lr_for_update(n),'current_row_lr':x.lr_for_update(n),
   'scheduled_next_update':n+1,'runtime_state':{'sampler':'opaque_native_frontier','RNG':('opaque_python','opaque_cpu','opaque_CUDA')}}
  before=copy.deepcopy(saved);runtime=x.load_phase_state(saved)
  check(saved==before and received[-1]['phase_identity']==before['phase_identity'],'own restore never mutates input/staticidentity')
  check(runtime==saved['runtime_state'] and runtime is not saved['runtime_state'] and x.optimizer_updates_completed==75
    and x._base.optimizer_updates_completed==575 and x.cumulative_step==7075 and x.last_applied_lr==x.lr_for_update(75),
    'returned opaque runtime/LOCAL75/Adam575/global7075 frontier restored')
  calls=len(received)
  for field,value in [('current_row_lr',1e-3),('last_applied_lr',1e-4),('scheduled_next_update',99),
                      ('scheduled_acquisition_updates',True),('continued_acquisition_updates',74),('kind',OLD_KIND)]:
   bad=copy.deepcopy(saved);bad[field]=value;reject(lambda b=bad:x.load_phase_state(b),'saved metadata corruption '+field)
  bad=copy.deepcopy(saved);bad['row_schedule']['warmup_updates']=25
  reject(lambda:x.load_phase_state(bad),'warmup25 rejects')
  check(len(received)==calls,'all metadata corruptions reject before inherited tensor-copy loader')
  def saver(x,*,runtime_state=None):return {'kind':OLD_KIND,'phase_identity':x.phase_identity(),
    'weights':{str(i):'opaque_tensor' for i in range(207)},'acquisition_optimizer':'opaque_three_states',
    'continued_acquisition_updates':x._local_updates,'acquisition_optimizer_updates':500+x._local_updates,
    'runtime_state':copy.deepcopy(runtime_state)}
  Parent.state_dict=saver
  own=x.state_dict(runtime_state=runtime)
  check(own['kind']==m.STATE_KIND and len(own['weights'])==207 and own['runtime_state']==runtime
    and own['scheduled_acquisition_updates']==75 and own['last_applied_lr']==x.current_row_lr,
    'own scheduled serializer decorates inherited full207/3Adam/runtime rather than filtering requiresgrad')
  # Saved LR is derived from K+N minus K, not the current live schedule point.
  x=stage(1e-3);previous=x._base.declared_optimizer_group
  def auditor(x,state,total):
   assert total==575 and x._base.declared_optimizer_group['lr']==x.lr_for_update(75)
   raise ValueError('mock moment auditor rejection')
  Parent._audit_optimizer=auditor
  optimizer={'param_groups':[{'params':[0,1,2],**x._static_group,'lr':x.lr_for_update(75)}]}
  reject(lambda:x._audit_optimizer(optimizer,575),'inherited strict moment auditor rejection propagates')
  check(x._base.declared_optimizer_group is previous and x.current_row_lr==1e-4,
        'temporary savedgroup audit restores live metadata without touching liveLR/moments')
 finally:
  Parent.load_phase_state=old_load;Parent.state_dict=old_save;Parent._audit_optimizer=old_audit
 check('torch' not in sys.modules and 'nemo' not in sys.modules,'no ML imports/tensors/checkpoints/GPU/processes')
 output=args.output.resolve();assert not output.exists(),'Fresh report path required'
 report={'schema':1,'kind':'BPE_scheduled_acquisition_stdlib_verification_v1','passed':True,
  'utility_sha256':hashlib.sha256(UTILITY.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'checks':checks,'checks_passed':len(checks),'rejections':rejections,'rejections_passed':len(rejections),
  'scope':'Actual pure schedule and class step/save-load dispatch with explicit optimizer/tensor-loader mocks. No actual model/source207/phase1 state or moments copied, native ASR, CPU-Torch, GPU, jobs, shared state or Backlog. Hardware/source-byte and complete real tensor restore remain ROOT gates.'}
 with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejections),'utility_sha256':report['utility_sha256'],'report':str(output)}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
