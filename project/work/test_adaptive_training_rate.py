"""Synthetic CPU sampler/source/audio/time corruption checks; no GPU use."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import verify_adaptive_results as worker
from train_lora_batched import BalancedBucketSampler

fixture=ROOT/'work/adaptive_training_rate_fixture';out=fixture/'outputs/parakeet_turkish';data=fixture/'work/data';run=out/'runs/fixture'
data.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def jsonl(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
def example(lang,i):
    return {'source_id':f'synthetic:{lang}:train:{i}','sentence_id':f'train:{lang}:{i}',
            'dataset_split':'train','language':lang,'text':f'unique training {lang} phrase {i}',
            'duration':.25*(i+1),'audio_sha256':f'audio:{lang}:{i}','wav_sha256':f'wav:{lang}:{i}',
            'encoder_frames':20+i,'target_tokens':10+i}
tr=[example('tr',i) for i in range(5)]
replay=[example(lang,i) for lang in worker.LANGUAGES[:-1] for i in range(3)]
trpath=data/'tr_train.jsonl';replaypath=data/'replay_train.jsonl'
jsonl(trpath,tr);jsonl(replaypath,replay)
for lang in worker.LANGUAGES:
    for split in ('validation','test'):
        jsonl(data/f'{split}_{lang}.jsonl',[{'source_id':f'heldout:{lang}:{split}',
              'sentence_id':f'heldout:{lang}:{split}','language':lang,'text':f'heldout distinct {lang} {split}',
              'audio_sha256':f'heldout-audio:{lang}:{split}','wav_sha256':f'heldout-wav:{lang}:{split}'}])
config={'run_id':'fixture','steps':20,'batch_size':1,'accumulation':4,'effective_batch':4,
        'tr_clips_per_update':2,'replay_clips_per_update':2,'seed':20261010,'replay_fraction':.5,'bucket_size':3,
        'config_identity':{key:hashlib.sha256('\n'.join(sorted(r['source_id'] for r in pool)).encode()).hexdigest()
                           for key,pool in [('tr_pool_sha256',tr),('replay_pool_sha256',replay)]},
        'input_sha256':{str(p.resolve()):worker.file_sha(p) for p in data.glob('*.jsonl')},
        'training_throughput_timing':'Synthetic timers only; no measured GPU training'}
pool={'turkish':tr,'replay':replay,'input_sha256':config['input_sha256']}
sampler=BalancedBucketSampler(tr,replay,seed=config['seed'],batch_size=1,accumulation=4,replay_fraction=.5,bucket_size=3)
audio=0.;language_audio={lang:0. for lang in worker.LANGUAGES};seen=set();replaycounts={lang:0 for lang in worker.LANGUAGES[:-1]};history=[]
for step in range(1,11):
    samples=[r for batch in sampler.next_batches() for r in batch];stepaudio=sum(r['duration'] for r in samples);audio+=stepaudio
    for row in samples:
        language_audio[row['language']]+=row['duration']
        if row['language']=='tr':seen.add(row['source_id'])
        else:replaycounts[row['language']]+=1
    wall=step*.5
    history.append({'step':step,'sampled_source_ids':[r['source_id'] for r in samples],
                    'step_audio_seen_seconds':stepaudio,'audio_seen_seconds':audio,'train_wall_seconds':wall,
                    'audio_minutes_per_wall_minute':audio/wall,'turkish_clips_seen':step*2,'replay_clips_seen':step*2,
                    'turkish_unique_clips_seen':len(seen),'replay_language_clip_counts':replaycounts.copy(),
                    'turkish_epoch':step*2/len(tr),'language_audio_seen_seconds':language_audio.copy(),
                    'turkish_audio_seen_seconds':language_audio['tr'],'replay_audio_seen_seconds':audio-language_audio['tr']})
for name,value in [('run_config.json',config),('training_pool.json',pool),('training_history.json',history)]:write(run/name,value)
worker.ROOT=fixture;worker.OUTPUT=out;worker.DATA=data
checks=worker.Checks();actual=worker.audit_training_rate('fixture',checks)
tests=[{'name':'exact seeded bucket sampler, clip/audio counters and synthetic timers accepted','passed':True,'checks':len(checks.items)}]
changes=[('different sampler seed',lambda c,p,h:c.__setitem__('seed',1)),
         ('different sample order',lambda c,p,h:h[0]['sampled_source_ids'].reverse()),
         ('wrong cumulative audio',lambda c,p,h:h[-1].__setitem__('audio_seen_seconds',0.)),
         ('wrong step audio',lambda c,p,h:h[0].__setitem__('step_audio_seen_seconds',.1)),
         ('wrong wall-time rate',lambda c,p,h:h[-1].__setitem__('train_wall_seconds',100.)),
         ('wrong unique TR count',lambda c,p,h:h[-1].__setitem__('turkish_unique_clips_seen',1)),
         ('wrong replay language counts',lambda c,p,h:h[0]['replay_language_clip_counts'].__setitem__('en',100)),
         ('wrong source reference',lambda c,p,h:p['turkish'][0].__setitem__('text','altered reference')),
         ('wrong source duration',lambda c,p,h:p['turkish'][0].__setitem__('duration',99.)),
         ('wrong accepted pool hash',lambda c,p,h:c['config_identity'].__setitem__('tr_pool_sha256','0'*64)),
         ('wrong epoch',lambda c,p,h:h[0].__setitem__('turkish_epoch',1.))]
for name,mutate in changes:
    c,p,h=copy.deepcopy(config),copy.deepcopy(pool),copy.deepcopy(history);mutate(c,p,h)
    for filename,value in [('run_config.json',c),('training_pool.json',p),('training_history.json',h)]:write(run/filename,value)
    try:worker.audit_training_rate('fixture',worker.Checks())
    except Exception as error:tests.append({'name':name,'passed':True,'failure':str(error)})
    else:tests.append({'name':name,'passed':False})
for name,value in [('run_config.json',config),('training_pool.json',pool),('training_history.json',history)]:write(run/name,value)
payload={'passed':all(t['passed'] for t in tests),'gpu_used':False,'scope':'Synthetic training-pool/source/sampler/audio/timer fixtures only; no actual adaptive-run speed result.',
         'auditor_sha256':worker.file_sha(OUTPUT/'verify_adaptive_results.py'),'test_count':len(tests),'tests':tests,
         'synthetic_completed_updates':actual['completed_updates'],'synthetic_audio_seconds':actual['audio_seconds']}
target=OUTPUT/'adaptive_training_rate_auditor_cpu_verification.json';target.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':payload['passed'],'tests':len(tests),'output':str(target)},indent=2))
raise SystemExit(0 if payload['passed'] else 1)
