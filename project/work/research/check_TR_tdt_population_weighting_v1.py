"""Focused scalar/vector stubs, not a Torch/native gradient or state test."""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import TR_tdt_population_weighting_v1 as candidate

checks=[];rejections=[]
def check(ok,name):
    if not ok:raise AssertionError(name)
    checks.append(name)
def reject(fn,name):
    try:fn()
    except ValueError:rejections.append(name);return
    raise AssertionError('Accepted corruption: '+name)
class V:
    dtype='scalar_stub_float64';device='explicit_stdlib_stub_CPU'
    def __init__(self,x):self.x=list(x)
    def _op(self,b,fn):
        y=b.x if isinstance(b,V) else [b]*len(self.x)
        return V([fn(x,z) for x,z in zip(self.x,y)])
    def __mul__(self,b):return self._op(b,lambda a,z:a*z)
    def __add__(self,b):return self._op(b,lambda a,z:a+z)
    def __sub__(self,b):return self._op(b,lambda a,z:a-z)
    def __getitem__(self,i):return V([self.x[j] for j in i])
    def mean(self):return V([math.fsum(self.x)/len(self.x)])
    def detach(self):return self
    def float(self):return self
    def cpu(self):return self
    def tolist(self):return self.x.copy()
class T:
    @staticmethod
    def tensor(x,**kwargs):return V(x)
    @staticmethod
    def stack(x):return V([v.x[0] for v in x])
@dataclass
class Objective:
    total:V;tdt:V;token_kl:V;duration_kl:V;feature_distillation:V;per_clip:dict;target_tokens:tuple
    def detached_metrics(self):return {'loss':self.total.x[0],'tdt_loss':self.tdt.x[0],'replay_kl':self.token_kl.x[0],'replay_duration_kl':self.duration_kl.x[0],'replay_feature_distillation':self.feature_distillation.x[0]}
u=[11,17,27,34,40,45,56,165,13,22,29,37,49,71,200,224]
languages=['tr']*14+['pt','de']
rows=[{'language':l,'target_tokens':n,'source_id':'fixture:'+str(i)} for i,(l,n) in enumerate(zip(languages,u))]
raw=[n*(.7+i*.013) for i,n in enumerate(u)]
preservation=[0.]*14+[40*.001+2*.002+20*.003,(40*.001+2*.002+20*.003)*5/3]
def original(indices,losses=raw,terms=preservation):
    tdt=V([losses[i]/u[i] for i in indices]);total=V([losses[i]/u[i]+terms[i] for i in indices])
    return Objective(total.mean(),tdt.mean(),V([.01]),V([.02]),V([.03]),{'tdt':tdt,'total':total,'token_kl':V([.001]*len(indices)),'duration_kl':V([.002]*len(indices)),'feature_distillation':V([.003]*len(indices))},tuple(u[i] for i in indices))
def accumulated(losses=raw,terms=preservation,mode='token_population'):
    total=0.;outputs=[]
    for indices in (list(range(8)),list(range(8,16))):
        source=original(indices,losses,terms)
        result,exposure=candidate.apply_to_objective(source,[rows[i] for i in indices],mode=mode,torch_module=T)
        total+=result.total.x[0]/2;outputs.append((source,result,exposure))
    return total,outputs
value,outputs=accumulated()
expected=math.fsum(raw[i]/candidate.POPULATION_MEAN_U if i<14 else raw[i]/u[i] for i in range(16))/16+math.fsum(preservation)/16
check(abs(value-expected)<1e-14,'8TR then6TR+2RP mean16 equals declared formula')
for j,(source,result,exposure) in enumerate(outputs):
    off,meta=candidate.apply_to_objective(source,rows[j*8:(j+1)*8],mode='clip')
    check(off is source,'off-mode exact original object micro'+str(j))
    check(meta['all_clip_weights']==[1.]*8,'off-mode all weights1 micro'+str(j))
    check(all(result.per_clip[k] is source.per_clip[k] for k in ('token_kl','duration_kl','feature_distillation')),'preservation vectors and graph identities retained micro'+str(j))
    check(source.per_clip['tdt'].x==[raw[i]/u[i] for i in range(j*8,(j+1)*8)],'original objective unmodified micro'+str(j))
    metrics=candidate.detached_metrics(result,exposure,torch_module=T)
    check(metrics['tdt_loss_unweighted']==source.tdt.x[0] and metrics['tdt_loss_weighted']==result.tdt.x[0],'both TDT metrics explicit micro'+str(j))
check(outputs[1][2]['all_clip_weights'][-2:]==[1.,1.],'PT200/DE224 positive RP lengths unaffected by TR180 cap')
check(outputs[1][1].per_clip['total'].x[-2:]==outputs[1][0].per_clip['total'].x[-2:],'every RP total literal unchanged')
check(outputs[1][1].per_clip['tdt'].x[-2:]==outputs[1][0].per_clip['tdt'].x[-2:],'every RP TDT literal unchanged')
delta=1e-5
max_gradient_error=0.
for i in range(16):
    plus=raw.copy();minus=raw.copy();plus[i]+=delta;minus[i]-=delta
    numerical=(accumulated(plus)[0]-accumulated(minus)[0])/(2*delta)
    exact=1/(16*(candidate.POPULATION_MEAN_U if i<14 else u[i]))
    max_gradient_error=max(max_gradient_error,abs(numerical-exact))
check(max_gradient_error<2e-11,'finite differences every TR/RP rawNLL coefficient')
for i in (14,15):
    plus=preservation.copy();minus=preservation.copy();plus[i]+=delta;minus[i]-=delta
    numerical=(accumulated(terms=plus)[0]-accumulated(terms=minus)[0])/(2*delta)
    check(abs(numerical-1/16)<2e-11,'RP preservation coefficient derivative unchanged '+str(i))
# Opaque state carrier is deliberately never offered to the utility: source
# objects/207 moments/history/queues/RNG stay byte-identical. Native own-state
# loading will remain the pinned worker's responsibility after actual choice.
rng=random.Random(4242)
state={'weights':{str(i):[i,0.] for i in range(207)},'optimizer':{'state':{i:{'step':25109,'exp_avg':[i*.01],'exp_avg_sq':[i*.001]} for i in range(207)}},'history':[{'step':i,'validation':{'passed':False}} for i in range(1,17)],'sampler':{'updates':25109,'order':['a','b'],'position':2,'RNG':rng.getstate()},'python_rng':rng.getstate(),'torch_rng':b'opaqueCPU','cuda_rng':[b'opaqueCUDA'],'selected_step':19750,'retention_failures':23,'step':25109}
before=pickle.dumps(state,protocol=5);before_rng=rng.getstate()
accumulated();candidate.declaration('token_population')
restored=pickle.loads(before)
check(before==pickle.dumps(state,protocol=5),'all207 state/history/moments/sampler/RNG untouched by candidate')
check(pickle.dumps(restored,protocol=5)==before,'opaque checkpoint roundtrip no resets')
check(rng.getstate()==before_rng,'independent runtime RNG unchanged')
def metadata_bad(n):
    r=copy.deepcopy(rows);lengths=u.copy();lengths[0]=n;r[0]['target_tokens']=n
    return lambda:candidate.weights_for_metadata(r,lengths,mode='token_population')
reject(metadata_bad(0),'empty TR target rejected without clamp')
reject(metadata_bad(-1),'negative target rejected')
reject(metadata_bad(True),'boolean target rejected')
reject(metadata_bad(181),'TR admission cap180 retained')
reject(lambda:candidate.weights_for_metadata(rows,u,mode='other'),'unsupported mode')
reject(lambda:candidate.weights_for_metadata(rows,u,mode='clip',population_mean_U=float('nan')),'nonfinite mean rejected')
reject(lambda:candidate.weights_for_metadata(rows,u,mode='token_population',population_mean_U=27.),'changed fixed mean rejected')
reject(lambda:candidate.apply_to_objective(original(list(range(8))),rows[:8],mode='token_population'),'missing injected runtime rejected')
rp_only=rows[14:];rp_lengths=u[14:]
check(candidate.weights_for_metadata(rp_only,rp_lengths,mode='clip')['all_clip_weights']==[1.,1.],'off-mode replay-only metadata delegates unchanged')
check('torch' not in sys.modules,'no Torch imported')
path=OUT/'TR_tdt_population_weighting_stdlib_CPU_verification_v1.json'
assert not path.exists()
source=OUT/'TR_tdt_population_weighting_v1.py'
report={'kind':'inactive_TR_tdt_population_weighting_stdlib_verification_v1','passed':True,'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size},'checks':checks,'rejections':rejections,'positive_count':len(checks),'rejection_count':len(rejections),'finite_difference_max_error':max_gradient_error,'fixture_precision':'Explicit stdlib float64 vector stubs; not Torch/autograd/native FP32 proof','checkpoint_fixture_scope':'Opaque207 Python value maps only; candidate never touches optimizer/history/sampler/RNG. No real saved checkpoint loading or state admission.','actual_GPU_tested':False,'head_policy_or_source_admitted':False,'Torch_imported':False}
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'passed':True,'source_sha256':report['source']['sha256'],'positive_count':len(checks),'rejection_count':len(rejections),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'finite_difference_max_error':max_gradient_error,'actualGPU':False}))
