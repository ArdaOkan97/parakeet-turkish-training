"""CPU checks of bucket coverage, exact-resume state and trainer guards."""
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import torch
from transformers import AutoConfig, ParakeetForTDT
from train_lora_batched import (BalancedBucketSampler, BucketCycler, lr_factor,
    next_evaluation_epoch, development_gate, source_isolation, validate_trainable_checkpoint,
    parser, validate_arguments)
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from train import checkpoint_state

torch.set_num_threads(1)
torch.manual_seed(20261010)
checks = []
def check(name,value):
    checks.append({'name':name,'passed':bool(value)})
    if not value:
        raise AssertionError(name)
def rejects(name,function):
    try:
        function()
    except (ValueError,KeyError):
        check(name,True)
    else:
        check(name,False)
def row(language,index):
    return {'source_id':f'cpu:{language}:{index}','language':language,'duration':1+index/10,
            'encoder_frames':7+index*3,'target_tokens':2+index,'text':f'example {language} {index}',
            'dataset_split':'train','audio_sha256':f'audio-{language}-{index}'}
tr = [row('tr',i) for i in range(13)]
replay = [row(lang,i) for lang in ('en','es','it','fr','de') for i in range(7)]
def sampler(batch=2,accum=2,fraction=.5):
    return BalancedBucketSampler(tr,replay,seed=20261010,batch_size=batch,
                                  accumulation=accum,replay_fraction=fraction,bucket_size=5)
def draw(s):
    groups=s.next_batches()
    return [[r['source_id'] for r in g] for g in groups]
all_sequences=[]
for b,a in ((1,4),(2,2),(4,1)):
    s=sampler(b,a)
    seq=[]
    for _ in range(300):
        groups=s.next_batches()
        check(f'B{b} update{_+1} shape',len(groups)==a and all(len(g)==b for g in groups))
        flattened=[r for g in groups for r in g]
        check(f'B{b} update{_+1} mix',sum(r['language']=='tr' for r in flattened)==2)
        seq.append([r['source_id'] for r in flattened])
    check(f'B{b} Turkish unique coverage',len(s.tr.seen)==13 and s.tr.draws==600)
    check(f'B{b} replay counts',sum(s.replay_language_counts.values())==600 and all(s.replay_language_counts[l]==v.draws for l,v in s.replay.items()))
    all_sequences.append(seq)
check('B1/B2/B4 same sampled source order',all_sequences[0]==all_sequences[1]==all_sequences[2])
s=sampler()
for _ in range(23): draw(s)
state=copy.deepcopy(s.state_dict())
path=ROOT/'work/batched_sampler_cpu_state.pt'
torch.save(state,path)
loaded=torch.load(path,map_location='cpu',weights_only=True)
r=sampler();r.load_state_dict(loaded)
for i in range(200):
    check(f'exact resume next update{i+1}',draw(s)==draw(r))
check('exact resume final full state',s.state_dict()==r.state_dict())
bad=copy.deepcopy(state);bad['tr']['draws']+=1
rejects('corrupt cycle draw counter rejected',lambda:sampler().load_state_dict(bad))
bad=copy.deepcopy(state);bad['replay_language_counts']['en']+=1
rejects('corrupt replay count rejected',lambda:sampler().load_state_dict(bad))
bad=copy.deepcopy(state);bad['tr']['position']=14
rejects('out-of-range cursor rejected',lambda:sampler().load_state_dict(bad))
rejects('nonintegral replay mix rejected',lambda:sampler(fraction=.3))
rejects('zero Turkish stream rejected',lambda:sampler(fraction=1))
rejects('zero target length rejected',lambda:BucketCycler([{**tr[0],'target_tokens':0}],random.Random(1)))
for fraction in (.25,.75):
    s=sampler(fraction=fraction)
    groups=s.next_batches()
    check(f'fraction{fraction} exact counts',sum(x['language']=='tr' for g in groups for x in g)==round(4*(1-fraction)) and all(len(g)==2 for g in groups))
s=BucketCycler(tr,random.Random(77),bucket_size=5)
for cycle in range(5):
    sources=[s.next()['source_id'] for _ in range(13)]
    check(f'cycle{cycle+1} complete unique pool',len(set(sources))==13 and set(sources)=={r['source_id'] for r in tr})
check('warmup first LR',lr_factor(1,100,25,'constant')==.04)
check('warmup last LR',lr_factor(25,100,25,'constant')==1)
check('constant after warmup',lr_factor(26,100,25,'constant')==1)
check('cosine final floor',math.isclose(lr_factor(100,100,25,'cosine'),.1,abs_tol=1e-12))
check('cosine monotonic after warmup',all(lr_factor(i,100,25,'cosine')>=lr_factor(i+1,100,25,'cosine') for i in range(25,100)))
check('quarter-half-full cadence',[next_evaluation_epoch(v,.25,1) for v in (0,.25,.5,1,2)]==[.25,.5,1,2,3])
check('overshoot cadence',[next_evaluation_epoch(v,.25,1) for v in (.3077,.6154,1.077)]==[.5,1,2])
check('twice-per-epoch cadence',next_evaluation_epoch(1,.25,2)==1.5 and next_evaluation_epoch(1.5,.25,2)==2)
baseline={f'validation_{l}':{'asr':{'wer':.02 if l!='tr' else 1}} for l in ('en','es','it','fr','de','tr')}
dev=copy.deepcopy(baseline);dev['validation_tr']['asr']['wer']=.4
dev['validation_en']['asr']['wer']=.025
check('exact retention boundary passes',development_gate(dev,baseline)['retention_gate_passed'])
dev['validation_en']['asr']['wer']=.025001
check('above retention boundary fails',not development_gate(dev,baseline)['retention_gate_passed'])
heldout=[{**row('tr',99),'dataset_split':'validation','sentence_id':123,'text':'Şeker güzel'}]
source_isolation(tr,replay,heldout);check('disjoint train accepted',True)
rejects('cross-language global sentence ID excluded',lambda:source_isolation([{**tr[0],'sentence_id':123}],replay,heldout))
rejects('normalized Turkish heldout reference excluded',lambda:source_isolation([{**tr[0],'text':'şeker güzel.'}],replay,heldout))
rejects('source leakage excluded',lambda:source_isolation([{**heldout[0],'dataset_split':'train'}],replay,heldout))
rejects('audio leakage excluded',lambda:source_isolation([{**tr[0],'audio_sha256':heldout[0]['audio_sha256']}],replay,heldout))
rejects('nontrain split excluded',lambda:source_isolation([{**tr[0],'dataset_split':'test'}],replay,heldout))
config=AutoConfig.from_pretrained((ROOT/'work/base_model_path.txt').read_text().strip(),local_files_only=True)
config.encoder_config.hidden_size=16;config.encoder_config.intermediate_size=32
config.encoder_config.num_attention_heads=4;config.encoder_config.num_key_value_heads=4
config.encoder_config.num_hidden_layers=2;config.encoder_config.subsampling_conv_channels=4
config.decoder_hidden_size=8;config.vocab_size=23;config.blank_token_id=16
model=ParakeetForTDT(config)
layout=install_encoder_lora(model,rank=2,alpha=4)
weights=checkpoint_state(model)
saved={'weights':weights,'lora_metadata':layout}
validate_trainable_checkpoint(model,saved,layout);check('exact trainable layout accepted',True)
bad=copy.deepcopy(saved);bad['weights'].pop(next(iter(bad['weights'])))
rejects('missing trainable key rejected',lambda:validate_trainable_checkpoint(model,bad,layout))
bad=copy.deepcopy(saved);bad['weights']['extra']=torch.zeros(1)
rejects('extra trainable key rejected',lambda:validate_trainable_checkpoint(model,bad,layout))
bad=copy.deepcopy(saved);key=next(iter(bad['weights']));bad['weights'][key]=bad['weights'][key].half()
rejects('FP16 warm-start rejected',lambda:validate_trainable_checkpoint(model,bad,layout))
bad=copy.deepcopy(saved);bad['weights'][key].flatten()[0]=float('nan')
rejects('nonfinite warm-start rejected',lambda:validate_trainable_checkpoint(model,bad,layout))
bad=copy.deepcopy(saved);bad['lora_metadata']['targets'][0]['rank']=8
rejects('different adapter layout rejected',lambda:validate_trainable_checkpoint(model,bad,layout))
args=validate_arguments(parser().parse_args(['--run-id','cpu_guard','--no-encoder-checkpoint']))
check('no encoder checkpoint CLI honored',args.no_encoder_checkpoint)
for name,flag,value in [('negative head LR','--head-lr','-1'),('NaN coefficient','--replay-kl','nan'),('zero bucket','--bucket-size','0'),('negative target WER','--target-turkish-wer','-1'),('zero subset','--train-limit','0')]:
    rejects(name,lambda f=flag,v=value:validate_arguments(parser().parse_args(['--run-id','cpu_guard',f,v])))
rejects('unsafe run path rejected',lambda:validate_arguments(parser().parse_args(['--run-id','../unsafe'])))
incoming_path=ROOT/'work/lora_20261010_01_best.pt'
incoming=torch.load(incoming_path,map_location='cpu',weights_only=True)
full_config=AutoConfig.from_pretrained((ROOT/'work/base_model_path.txt').read_text().strip(),local_files_only=True)
full_config.vocab_size=8199
with torch.device('meta'):
    full_model=ParakeetForTDT(full_config)
    full_layout=install_encoder_lora(full_model,rank=8,alpha=16)
validate_trainable_checkpoint(full_model,incoming,full_layout)
check('actual1200 checkpoint exact full architecture shape/layout accepted',True)
check('actual1200 checkpoint all207 tensors',len(incoming['weights'])==207)
check('actual1200 checkpoint source run/step',incoming['run_id']=='lora_20261010_01' and incoming['step']==1200)
check('actual1200 checkpoint19M trainable values',sum(x.numel() for x in incoming['weights'].values())==19709068)
result={'schema':1,'passed':all(x['passed'] for x in checks),'gpu_used':False,
        'scope':'CPU sampler/coverage/exact serialized RNG-state replay, LR/cadence/gate/provenance and actual reduced model checkpoint guards; no full trainer GPU run',
        'check_count':len(checks),'checks':checks,'trainable_fixture_tensors':len(weights),
        'sampler_updates_per_batch_layout':300,'exact_resume_following_updates':200,
        'sampler_fixture_pools':{'tr':13,'replay':35,'replay_per_language':7},
        'actual_incoming_checkpoint':{'path':str(incoming_path),'sha256':hashlib.sha256(incoming_path.read_bytes()).hexdigest(),
                                      'run_id':incoming['run_id'],'step':incoming['step'],'tensor_count':len(incoming['weights']),
                                      'full_model_storage':'meta only; source checkpoint read on CPU; no base weights/GPU allocated'},
        'trainer_sha256':hashlib.sha256((OUT/'train_lora_batched.py').read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256((OUT/'batched_lora.py').read_bytes()).hexdigest(),
        'known_limit':'True B2/B4 real BF16 GPU equivalence remains failed/unapproved; passing CPU control-path checks does not authorize adopting batching.'}
(OUT/'batched_trainer_cpu_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('passed','check_count','trainable_fixture_tensors')}))
