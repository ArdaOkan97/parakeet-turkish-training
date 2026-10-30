"""One bounded producer-shaped source fixture; no DB/Torch/GPU/state migration."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'outputs/parakeet_turkish'
path=HERE/'track_continuation_v19.py';old=HERE/'track_continuation_v18.py'
tree=ast.parse(path.read_text());previous=ast.parse(old.read_text());checks=[]
def ok(value,label):
    assert value,label;checks.append(label)
ns={'Path':Path,'ROOT':ROOT,'json':json,'hashlib':hashlib,'math':math,'re':__import__('re')}
def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (ValueError,OSError):return default
ns['read_json']=read_json
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        try:ns[node.targets[0].id]=ast.literal_eval(node.value)
        except (ValueError,TypeError):pass
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
exempt={'follow','parser','main','publish_cycle_memory','start_controls'}
ok(all(ast.dump(n,include_attributes=False)==ast.dump(next(x for x in functions if x.name==n.name),include_attributes=False)
       for n in previous.body if isinstance(n,ast.FunctionDef) and n.name not in exempt),'All inherited V18 math/function ASTs unchanged')
original_tree=ast.parse((HERE/'track_experiment.py').read_text());metrics=next(n for n in original_tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics_for')
original_ns={};exec(compile(ast.Module(body=[metrics],type_ignores=[]),'<original metrics>','exec'),original_ns)
ns['original']=SimpleNamespace(metrics_for=original_ns['metrics_for'])
config=read_json(HERE/'run_config.json');ok(config['phase']==ns['ENCODER_SCHEDULE_PHASE'],'Actual published encoder phase read')
# Parent V18 contexts were already applied to this immutable history before the
# new phase. Stub only inherited context/value work; actual new admission reads
# the real small receipt/config records and validates every producer field.
ns['full_epoch_phase_contexts_for']=lambda config,history:{'fixture_parent_only':True}
ns['full_epoch_phase_value_for_record']=lambda record,contexts,development:{'train/tdt_loss':record.get('tdt_loss',1.),'dev/wer_tr_percent':99.}
base={'step':25109,'audio_seen_seconds':1000.,'train_wall_seconds':500.,'turkish_audio_seen_seconds':700.,'replay_audio_seen_seconds':300.,
      'replay_language_clip_counts':{'en':1,'es':1,'it':1,'fr':1,'de':1,'pt':1}}
context=ns['encoder_context'](config,[base]);ok(context['step']==25109 and context['mode']=='cosine','Actual small receipt/config strict admission passed with synthetic counters')
development={25109:{'dev/wer_'+lang+'_percent':1. for lang in ('tr','en','es','it','fr','de','pt')}}
values=ns['encoder_phase_value_for_record'](base,{'encoder_schedule':context},development)
ok(values['retention/source_H_all_six_qualified']==0 and values['retention/all_six_passed']==0,'Actual source Spanish53 false retained')
ok(ns['encoder_phase_value_for_record']({**base,'step':25108},{'encoder_schedule':context},development).get('dev/wer_tr_percent') is None,'Inherited point never pretends new DEV')
def row(n):
    lr=1e-4 if n<=1 else 3e-5 if n==18129 else 3e-5+(1e-4-3e-5)*.5*(1+math.cos(math.pi*(n-1)/18128))
    return {**base,'step':25109+n,'tdt_loss':1.,'replay_kl':.01,'gradient_norm':1.,'elapsed_seconds':1.,'peak_allocated_vram_gb':6.,
        'audio_seen_seconds':1000.+20*n,'train_wall_seconds':500.+n,'turkish_audio_seen_seconds':700.+14*n,'replay_audio_seen_seconds':300.+6*n,
        'replay_language_clip_counts':{'en':1+n,'es':1,'it':1,'fr':1,'de':1,'pt':1+n},
        'encoder_schedule_updates_completed':n,'encoder_lr_schedule':'cosine','encoder_schedule_horizon':18129,
        'encoder_peak_lr':1e-4,'encoder_min_lr':3e-5,'head_lr':1e-5,'lora_lr':lr,'retention_stop_policy':'monitor_only',
        'preservation_coefficients_by_language':{'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]},
        'Turkish_domain_coverage':{'matched_cycle':{'source_step':25109,'local_updates':n,'cycle_index':2,'carried_cycle_visits':2,
            'new_cycle_visits':min(14*n,253796),'cycle_unique':2+min(14*n,253796),'full_cycle_complete':14*n>=253796,'tail_draws':max(0,14*n-253796)}}}
for n in (1,250,18129):
    result=ns['encoder_phase_value_for_record'](row(n),{'encoder_schedule':context},{})
    ok(result['phase/encoder_schedule_updates_completed']==n and result['resources/post_encoder_schedule_audio_minutes_per_wall_minute']==20.,'New local counter/rate and cosine endpoint')
    ok(result['data/matched_cycle_carried_visits']==2 and result['data/matched_cycle_tail_draws']==max(0,14*n-253796),'Cycle carry/tail distinct from new visits')
constant=deepcopy(context);constant['mode']='constant';record=row(250);record.update(encoder_lr_schedule='constant',lora_lr=1e-4)
ok(ns['encoder_phase_value_for_record'](record,{'encoder_schedule':constant},{})['train/lora_lr']==1e-4,'Constant counterpart fields supported')
for field,value in (('lora_lr',3e-4),('encoder_schedule_updates_completed',25110)):
    corrupt=row(1);corrupt[field]=value
    try:ns['encoder_phase_value_for_record'](corrupt,{'encoder_schedule':context},{})
    except ValueError:checks.append('Corrupt '+field+' refused')
    else:raise AssertionError(field)
ok(not ns['encoder_rejected_run_terminal'](config,{'completed_steps':25200,'full_new_uniform_pass_complete':True}),'Inherited fullpass never finishes new budget window')
ok(ns['encoder_rejected_run_terminal'](config,{'completed_steps':43238,'Turkish_domain_coverage':{'matched_cycle':{'full_cycle_complete':True}}}),'Only own actual complete new cycle can finish')
ok(ns['SOURCE_V18_SHA']==hashlib.sha256(old.read_bytes()).hexdigest(),'Migration pins exact CLOSED V18 producer')
proof={'schema':1,'kind':'Trackio_V19_encoder_schedule_focused_stdlib_verification_v1','passed':True,'checks':len(checks),'assertions':checks,
       'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Actual small published config/byte-bound receipt/parent config read; inherited context builder and counters mocked; pure field math/legacy AST checks. No whole history/state/mirror/DB/Torch/GPU reads or writes.'}
out=HERE/'Trackio_V19_encoder_schedule_focused_stdlib_verification_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'checks':len(checks),'source_sha256':proof['source_sha256'],'proof_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
