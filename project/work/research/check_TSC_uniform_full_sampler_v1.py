"""Tiny actual stdlib sampler migration/restore; synthetic TRAIN rows only."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(HERE))
from tsc_domain_sampler_v3 import TSCDomainBalancedBucketSampler,TSC_REPO,TSC_REVISION
from tsc_uniform_full_sampler_v1 import TSCUniformFullSampler,canonical
from turkish_domain_sampler_v1 import FLEURS_REVISION,CV_REVISION
tree=ast.parse((HERE/'train_lora_batched_data_v11.py').read_text(encoding='utf-8'))
nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('ids_sha','BucketCycler')]
ns={'hashlib':hashlib,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen BucketCycler AST','exec'),ns)
cycler=ns['BucketCycler'];checks=[];rejected=[]
def ok(value,message):
    if not value:raise AssertionError(message)
    checks.append(message)
def bad(call,message):
    try:call()
    except (ValueError,KeyError):checks.append(message);rejected.append(message);return
    raise AssertionError('unexpected acceptance '+message)
def row(source,language='tr',domain='cv'):
    dataset='google/fleurs' if domain=='fleurs' else TSC_REPO if domain=='tsc' else 'ysdede/commonvoice_17_tr_fixed'
    revision=FLEURS_REVISION if domain=='fleurs' else TSC_REVISION if domain=='tsc' else CV_REVISION
    return {'source_id':source,'language':language,'dataset':dataset,'dataset_revision':revision,
        'dataset_config':'tr_tr' if language=='tr' else 'pt_br' if language=='pt' else language,
        'dataset_split':'train','training_role':'train','text':'tiny training text','duration':1.,
        'encoder_frames':10+len(source),'target_tokens':3,'audio_sha256':'0'*64,'label_ids_sha256':'1'*64}
tr=[row('F'+str(i),domain='fleurs') for i in range(3)]+[row('C'+str(i)) for i in range(2)]+[row('T'+str(i),domain='tsc') for i in range(5)]
rp=[row(lang+str(i),language=lang,domain='fleurs') for lang in ('en','es','it','fr','de') for i in range(2)]
pt=[row('pt'+str(i),language='pt',domain='fleurs') for i in range(2)]
kwargs={'seed':11,'replay_seed':22,'batch_size':8,'accumulation':2,'bucket_size':4,'replay_fraction':.5,'fleurs_fraction':.3,
    'cycler_factory':cycler,'cycler_source_sha256':'2'*64}
old=TSCDomainBalancedBucketSampler(tr,rp,sampling_mode='domain30',**kwargs)
for _ in range(5):old.next_batches()
source=old.state_dict();before=canonical(source)
def fresh():return TSCUniformFullSampler(tr,rp,portuguese_replay=pt,portuguese_seed=33,sampling_mode='uniform',**kwargs)
new=fresh();evidence=new.migrate_source_state(source,old.identity)
ok(canonical(source)==before,'source sampler bytes/metadata not mutated')
ok(new.updates==5 and new.tr_draws==40 and new.tr_unique==old.tr_unique,'historical counters and seen IDs carried')
ok(new.turkish_epoch==0 and new.turkish_streams['uniform'].draws==0,'new uniform coverage epoch starts zero')
ok(new.tr_rng.getstate()==source['turkish_rng'] and new.replay_rng.getstate()==source['replay_rng'],'both source sampler RNGs copied')
for language in ('en','es','it','fr','de'):
    ok(new.replay[language].state_dict()==source['replay'][language],'original '+language+' queue native state exact')
ok(new.replay_language_counts=={**source['replay_language_counts'],'pt':0} and new.replay['pt'].draws==0,'Portuguese starts fresh without re-presenting old queue prefix')
ok(evidence['old_five_future_replay_sequence_claimed'] is False,'six-language selection nuisance explicit')
drawn=[]
for _ in range(new.uniform_pass_updates):drawn.extend(row['source_id'] for row in new.next_batches()[0])
ok(len(set(drawn[:len(tr)]))==len(tr) and set(drawn[:len(tr)])=={row['source_id'] for row in tr},'first uniform cycle visits every clip exactly once')
ok(len(drawn)-len(tr)==evidence['last_full_batch_extra_next_cycle_repeats'],'final whole batch repeats only declared next-cycle tail')
ok(new.coverage()['uniform_pass']['full_pass_complete'] and new.updates==7,'actual new full unique coverage and cumulative update clock')
ok(new.tr_draws==new.updates*8 and sum(stream.draws for stream in new.replay.values())==new.updates*8,'cumulative TR/replay totals retain whole B8x2')
saved=new.state_dict();restored=fresh();restored.migrate_source_state(source,old.identity);restored.load_state_dict(saved)
ok(restored.state_dict()==saved,'own uniform complete sampler exact restore')
for index in range(6):
    a=[[row['source_id'] for row in group] for group in new.next_batches()]
    b=[[row['source_id'] for row in group] for group in restored.next_batches()]
    ok(a==b,'future own uniform/PT draw sequence exact '+str(index))
for key in ('updates','replay_language_counts','domain_draws','source_group_draws'):
    changed=copy.deepcopy(saved)
    if key=='updates':changed[key]+=1
    else:changed[key][next(iter(changed[key]))]+=1
    target=fresh();target.migrate_source_state(source,old.identity)
    bad(lambda state=changed, sampler=target:sampler.load_state_dict(state),'corrupt own '+key+' rejected')
target=fresh();bad(lambda:target.load_state_dict(source),'old domain state cannot masquerade as uniform exact')
changed=copy.deepcopy(source);changed['turkish_streams']['fleurs']['position']+=1
bad(lambda:fresh().migrate_source_state(changed,old.identity),'invalid legacy queue boundary rejected')
changed=copy.deepcopy(source);changed['source_group_seen']['tsc']=[]
bad(lambda:fresh().migrate_source_state(changed,old.identity),'historical source coverage erasure rejected')
ok('torch' not in sys.modules and 'trackio' not in sys.modules,'no ML or mirror imported')
report={'schema':1,'kind':'inactive_TSC_uniform_full_sampler_tiny_CPU_verification_v1','passed':True,'checks':checks,
    'assertions':len(checks),'rejections':len(rejected),'sampler_sha256':hashlib.sha256((HERE/'tsc_uniform_full_sampler_v1.py').read_bytes()).hexdigest(),
    'scope':'Actual sampler methods + frozen BucketCycler AST on ten synthetic TRAIN rows and tiny six-language replay; no real corpus/PT/checkpoint/ML/GPU/source admission.',
    'actual_full_pool_loaded':False,'actual_source_admitted':False}
output=HERE/'TSC_uniform_full_sampler_tiny_CPU_verification_v1.json'
with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejected),'sampler_sha256':report['sampler_sha256'],
    'report_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
