from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import verify_large_batch_results as audit

tests=[]

def run(name,function,expected=True):
    try:
        result=function()
        passed=expected
        detail={'checks':len(result.items)} if isinstance(result,audit.Checks) else {}
    except (AssertionError,ValueError,KeyError,RuntimeError) as exc:
        passed=not expected
        detail={'failure':str(exc)}
    tests.append({'name':name,'passed':passed,**detail})
    if not passed:raise AssertionError(name)

torch.manual_seed(123)
torch.set_num_threads(1)
snapshot={'weights':{'decoder.embedding.weight':torch.randn(11,3),
    'joint.head.weight':torch.randn(16,3),'joint.head.bias':torch.randn(16),
    'encoder.layers.0.attention.lora_A':torch.randn(3,3)}}
gradients={k:torch.randn_like(v)*7 for k,v in snapshot['weights'].items()}
identity={'source_ids':[f's{i}' for i in range(16)],'precision':'bf16','weights_sha256':'fixture-only'}
evidence={'identity':identity,'gradients':gradients,
          'encoder_features':[torch.randn(1,i+2,3) for i in range(16)],
          'components':[2.,.05,.02,.1]}
update={'optimizer':'AdamW','initial_state_empty':True,'old_size':8,'vocab_size':11,
        'row_multiplier':15.,'max_grad_norm':1.,'lr':2e-4,'betas':[.9,.999],
        'eps':1e-8,'weight_decay':.01,'parameters':{}}
parameters={k:torch.nn.Parameter(v.clone()) for k,v in snapshot['weights'].items()}
for name,p in parameters.items():p.grad=gradients[name].clone()
optimizer=torch.optim.AdamW(list(parameters.values()),lr=update['lr'],betas=tuple(update['betas']),
                           eps=update['eps'],weight_decay=update['weight_decay'],foreach=False)
update['unclipped_global_norm']=float(torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,foreach=False))
optimizer.step()
for name in ('decoder.embedding.weight','joint.head.weight','joint.head.bias'):
    before=snapshot['weights'][name].clone()
    after=parameters[name].detach().clone()
    final=after.clone();final[8:11]=before[8:11]+15*(after[8:11]-before[8:11])
    update['parameters'][name]={'before':before,'after_adam':after,'after_multiplier':final,
        'unclipped_gradient':gradients[name].clone(),'clipped_gradient':parameters[name].grad.clone(),
        'optimizer_step':1.,'exp_avg':optimizer.state[parameters[name]]['exp_avg'].clone(),
        'exp_avg_sq':optimizer.state[parameters[name]]['exp_avg_sq'].clone()}
evidence['adam_update']=update

def step(value):
    check=audit.Checks();audit.adam_update_audit(value,snapshot,check,'fixture');return check

run('fresh clipped AdamW step and bounded new-token multiplier',lambda:step(evidence))
def norm_independent_of_cpu_fp32_kernel():
    original=torch.nn.utils.clip_grad_norm_
    def forbidden(*args,**kwargs):
        raise RuntimeError('CPU FP32 reduction must not substitute for accurate saved-gradient norm')
    torch.nn.utils.clip_grad_norm_=forbidden
    try:return step(evidence)
    finally:torch.nn.utils.clip_grad_norm_=original
run('CPU norm audit uses independent float64 sum and FP32 coefficient semantics',norm_independent_of_cpu_fp32_kernel)
for name,mutation in [
    ('duration rows accidentally scaled',lambda x:x['adam_update']['parameters']['joint.head.bias']['after_multiplier'][11:].add_(.02)),
    ('old token rows accidentally scaled',lambda x:x['adam_update']['parameters']['joint.head.weight']['after_multiplier'][:8].add_(.02)),
    ('multiplier applied to entire embedding',lambda x:x['adam_update']['parameters']['decoder.embedding.weight']['after_multiplier'][:8].add_(.02)),
    ('wrong multiplier metadata',lambda x:x['adam_update'].update(row_multiplier=10.)),
    ('nonfresh Adam state',lambda x:x['adam_update'].update(initial_state_empty=False)),
    ('wrong clipping norm',lambda x:x['adam_update'].update(unclipped_global_norm=123.)),
    ('wrong Adam learning rate',lambda x:x['adam_update'].update(lr=.01)),
    ('NaN gradient',lambda x:x['gradients']['joint.head.bias'].__setitem__(0,float('nan'))),
    ('wrong Adam moment',lambda x:x['adam_update']['parameters']['joint.head.bias']['exp_avg'].add_(.03)),
    ('wrong Adam step',lambda x:x['adam_update']['parameters']['joint.head.weight'].update(optimizer_step=2.)),
    ('wrong clipped gradient',lambda x:x['adam_update']['parameters']['decoder.embedding.weight']['clipped_gradient'].add_(.02)),
]:
    value=copy.deepcopy(evidence);mutation(value)
    run(name,lambda value=value:step(value),False)

def compare(a,b,same_shape=True):
    check=audit.Checks();result=audit.correctness_comparison(a,b,check,'fixture',same_shape=same_shape)
    return result

result=compare(evidence,evidence)
run('all finite identical gradient/features pass same-shape gate',lambda:None if result['strict_gate']['passed'] else (_ for _ in ()).throw(AssertionError()))
candidate=copy.deepcopy(evidence)
candidate['encoder_features'][0][0,0,0]+=1e-7
result=compare(evidence,candidate)
run('one-ULP feature change fails same-shape bitwise gate',lambda:None if not result['strict_gate']['passed'] else (_ for _ in ()).throw(AssertionError()))
candidate=copy.deepcopy(evidence)
candidate['components']=[x+2e-5 for x in candidate['components']]
result=compare(evidence,candidate,False)
run('weighted objective has its own strict bound',lambda:None if not result['strict_gate']['passed'] else (_ for _ in ()).throw(AssertionError()))
candidate=copy.deepcopy(evidence);candidate['identity']['source_ids'].reverse()
run('different ordered training IDs rejected',lambda:compare(evidence,candidate),False)
candidate=copy.deepcopy(evidence);candidate['gradients']['joint.head.bias'][0]=float('nan')
run('nonfinite comparison gradient rejected',lambda:compare(evidence,candidate),False)

seconds=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
timing={'updates':10,'warmup_updates':2,'seconds':seconds,'median_seconds_per_update':5.5,
        'p90_seconds_per_update':10.,'audio_min_per_wall_min':320/55,
        'clips_per_second':160/55,'peak_allocated_vram_gb':3.,'peak_reserved_vram_gb':4.}
def timecheck(value):
    check=audit.Checks();audit.timing_audit(value,32.,check,'fixture');return check
run('timing means/midpoint median/p90/audio arithmetic accepted',lambda:timecheck(timing))
for name,key,value in [('median substituted for summed runtime','audio_min_per_wall_min',32/5.5+.1),
                       ('wrong per-update clip count','clips_per_second',4*10/55),
                       ('warmup counted as measured updates','updates',12),
                       ('reserved memory below allocation','peak_reserved_vram_gb',2.)]:
    changed=copy.deepcopy(timing);changed[key]=value
    run(name,lambda changed=changed:timecheck(changed),False)

rows=[];items=[]
for i in range(16):
    mel=i+17;frames=(mel+7)//8;targets=i%5+2
    rows.append({'language':'tr' if i%2==0 else ('en','es','it','fr','de')[(i//2)%5],
                 'duration':mel*.01})
    items.append({'input_frames':mel,'frame_length':frames,'target_length':targets,
                  'labels':torch.arange(targets,dtype=torch.int64)+100+i,
                  'features':torch.randn(frames,1024)})

def mask_fixture(batch_size):
    if batch_size==16:groups=[list(range(16))]
    elif batch_size==1:groups=[[i] for i in range(16)]
    else:groups=[group for start in range(0,8,batch_size) for group in
                 (list(range(0,16,2))[start:start+batch_size],list(range(1,16,2))[start:start+batch_size])]
    result=[]
    for indices in groups:
        ts=[items[i]['frame_length'] for i in indices];us=[items[i]['target_length'] for i in indices]
        lengths=[items[i]['input_frames'] for i in indices]
        labels=torch.full((batch_size,max(us)),8192,dtype=torch.int64)
        teacher=torch.zeros(batch_size,max(ts),1024)
        mask=torch.arange(max(lengths))[None,:]<torch.tensor(lengths)[:,None]
        mel=torch.randn(batch_size,max(lengths),128).masked_fill(~mask.unsqueeze(-1),0)
        for j,i in enumerate(indices):
            labels[j,:us[j]]=items[i]['labels'];teacher[j,:ts[j]]=items[i]['features']
        result.append({'indices':indices,'labels':labels,'frame_lengths':torch.tensor(ts,dtype=torch.int64),
            'target_lengths':torch.tensor(us,dtype=torch.int64),'decoder_input_ids':torch.cat((torch.full((batch_size,1),8192,dtype=torch.int64),labels),1),
            'input_attention_mask':mask,'input_features':mel,'base_features':teacher,
            'replay_indices':[j for j,i in enumerate(indices) if rows[i]['language']!='tr']})
    return result

def masks(value,batch_size=16):
    check=audit.Checks();audit.mask_audit(value,rows,items,batch_size,check,'fixture');return check
for b in (1,2,4,8,16):
    value=mask_fixture(b)
    run('padded effective16 B'+str(b)+' labels/masks/teacher/replay accepted',lambda value=value,b=b:masks(value,b))
for name,mutation in [
    ('nonblank target padding',lambda x:x[0]['labels'].__setitem__((0,-1),0)),
    ('wrong decoder prefix',lambda x:x[0]['decoder_input_ids'].__setitem__((0,0),0)),
    ('wrong cached frame length',lambda x:x[0]['frame_lengths'].__setitem__(0,4)),
    ('fractional input mask storage',lambda x:x[0].update(input_attention_mask=x[0]['input_attention_mask'].float())),
    ('nonzero mel padding',lambda x:x[0]['input_features'].__setitem__((0,-1,0),1.)),
    ('nonzero teacher padding',lambda x:x[0]['base_features'].__setitem__((0,-1,0),1.)),
    ('altered valid teacher feature',lambda x:x[0]['base_features'].__setitem__((0,0,0),100.)),
    ('Turkish entered teacher replay indexing',lambda x:x[0]['replay_indices'].append(0)),
    ('duplicate microbatch source',lambda x:x[0]['indices'].__setitem__(0,1)),
    ('incorrect actual WAV mask length',lambda x:x[0]['input_attention_mask'].__setitem__((0,16),False)),
]:
    value=mask_fixture(16);mutation(value)
    run(name,lambda value=value:masks(value),False)

per_clip={'tdt':[2.]*16,'token_kl':[0. if i%2==0 else .1 for i in range(16)],
    'duration_kl':[0. if i%2==0 else .02 for i in range(16)],
    'feature_distillation':[0. if i%2==0 else .03 for i in range(16)]}
per_clip['total']=[sum(c*per_clip[k][i] for c,k in zip(audit.COEFFICIENTS,audit.COMPONENTS)) for i in range(16)]
per_evidence={'per_clip':per_clip,'components':[sum(per_clip[k])/16 for k in audit.COMPONENTS]}
def percheck(value):
    check=audit.Checks();audit.per_clip_audit(value,rows,check,'fixture');return check
run('preservation mean includes zero Turkish terms in denominator16',lambda:percheck(per_evidence))
value=copy.deepcopy(per_evidence);value['components'][1]*=2
run('replay-only denominator8 is rejected',lambda:percheck(value),False)
value=copy.deepcopy(per_evidence);value['per_clip']['token_kl'][0]=.1
value['per_clip']['total'][0]+=.5;value['components'][1]+=.1/16
run('nonzero Turkish teacher loss rejected',lambda:percheck(value),False)

fixture=ROOT/'work/large_batch_auditor_cpu_fixture';fixture.mkdir(parents=True,exist_ok=True)
evidence_path=fixture/'metadata_evidence.pt';torch.save({'synthetic':True},evidence_path)
sources=('profile_large_batches.py','profile_batched_training.py','diagnose_batch_objective_math.py','batched_lora.py',
    'efficient_tdt.py','efficient_tdt_v2.py','runtime.py','encoder_lora.py','extend_vocab.py','train_lora.py','train.py','train_lora_batched.py')
meta={'case_id':'fixture_only','family':'typical','mode':'timing','batch_size':2,'precision':'bf16',
      'backend':'default','status':'measured','tdt_implementation':'v2','effective_batch':16,'accumulation':8,
      'encoder_checkpointing':True,'chunk_size':16,'source_sha256':{n:audit.sha(HERE/n) for n in sources},
      'objective_coefficients':dict(zip(audit.COMPONENTS,audit.COEFFICIENTS)),'turkish_preservation_terms_zero':True,
      'gpu_context':'solo','adoption_eligible':False,'backend_flags':{'tf32_matmul_allowed':False,'tf32_cudnn_allowed':False,
         'cudnn_benchmark':False,'sdpa_math_enabled':True,'sdpa_flash_enabled':True,'sdpa_mem_efficient_enabled':True,'sdpa_cudnn_enabled':True},
      'evidence_path':str(evidence_path),'evidence_sha256':audit.sha(evidence_path)}
def metacheck(value,wrong_hash=False):
    path=fixture/'fixture_only.json';path.write_text(json.dumps(value),encoding='utf-8')
    entry={k:value[k] for k in ('case_id','family','mode','batch_size','precision','backend','status')}
    entry.update(tdt=value['tdt_implementation'],report_sha256='0'*64 if wrong_hash else audit.sha(path),
                 evidence_sha256=value.get('evidence_sha256'),evidence_path=value.get('evidence_path'))
    check=audit.Checks();audit.metadata_audit(path,entry,check);return check
run('complete synthetic metadata binds actual frozen code/evidence hashes',lambda:metacheck(meta))
run('wrong report hash rejected',lambda:metacheck(meta,True),False)
for name,mutation in [
    ('effective32 mislabeled effective16',lambda x:x.update(effective_batch=32)),
    ('automatic adoption flag',lambda x:x.update(adoption_eligible=True)),
    ('TF32 enabled against declared control',lambda x:x['backend_flags'].update(tf32_matmul_allowed=True)),
    ('concurrent compute timing context',lambda x:x.update(gpu_context='concurrent')),
    ('replay KL coefficient changed',lambda x:x['objective_coefficients'].update(token_kl=.5)),
    ('wrong frozen source fingerprint',lambda x:x['source_sha256'].update(**{'runtime.py':'0'*64})),
    ('wrong evidence fingerprint',lambda x:x.update(evidence_sha256='0'*64)),
    ('BF16 timing mislabeled FP32',lambda x:x.update(precision='fp32')),
]:
    value=copy.deepcopy(meta);mutation(value)
    run(name,lambda value=value:metacheck(value),False)

stats=audit.timing_audit(timing,32.,audit.Checks(),'fixture')
base={'case_id':'fixture_B1','family':'typical','batch_size':1,'accumulation':16,'mode':'timing','precision':'bf16','backend':'default',
    'status':'measured_with_numerical_gate_failure','timing':{'end_to_end':stats},'identity':identity,'optimizer':{'head_lr':2e-5},
    'hardware':'synthetic_fixture','torch_version':'synthetic_fixture','initial_trainable_sha256':'fixture',
    'objective_coefficients':dict(zip(audit.COMPONENTS,audit.COEFFICIENTS)),'masking':{'lattice_padding_factor':1.},
    'numerical_comparison':{'strict_gate':{'passed':False}}}
other=copy.deepcopy(base);other.update(case_id='fixture_B2',batch_size=2,accumulation=8)
for key in ('median_seconds_per_update','p90_seconds_per_update','total_measured_seconds'):other['timing']['end_to_end'][key]*=.5
for key in ('audio_min_per_wall_min','clips_per_second'):other['timing']['end_to_end'][key]*=2
fp32={'case_id':'fixture_fp32_B2','family':'typical','batch_size':2,'mode':'fp32-parity','backend':'math','timing':{},
       'numerical_comparison':{'strict_gate':{'passed':True}}}
def tablecheck(value):
    check=audit.Checks();result=audit.comparison_table(value,check)
    assert result[0]['same_shape_bf16_gate_passed'] is False
    assert result[1]['gain_vs_matched_b1']==2 and result[1]['adoption_eligible'] is False
    assert result[1]['same_shape_bf16_gate_passed'] is False and result[1]['fp32_math_oracle_gate_passed'] is True
    return check
run('matched speedup preserves failed B1/B2 BF16 gates and no adoption',lambda:tablecheck([base,other,fp32]))
for name,mutation in [
    ('timing optimizer changed across batch sizes',lambda x:x['optimizer'].update(head_lr=1e-3)),
    ('timing checkpoint changed',lambda x:x.update(initial_trainable_sha256='different')),
    ('timing warmup counts differ',lambda x:x['timing']['end_to_end'].update(warmup_updates=3)),
    ('timing update counts differ',lambda x:x['timing']['end_to_end'].update(updates=11)),
    ('timing backend controls differ',lambda x:x.update(backend='math')),
    ('timing precision controls differ',lambda x:x.update(precision='fp32')),
]:
    value=copy.deepcopy(other);mutation(value)
    run(name,lambda value=value:tablecheck([base,value,fp32]),False)

coverage_table=[{'family':family,'batch_size':b,'precision':'bf16','backend':'default','timing_scope':'end_to_end'}
                for family in ('typical','long') for b in (1,2,4,8,16)]
run('full matched typical/long coverage accepted',lambda:None if audit.coverage_from_table(coverage_table,[])[1] else (_ for _ in ()).throw(AssertionError()))
changed=copy.deepcopy(coverage_table);changed[1]['backend']='math'
run('mixing attention backends cannot manufacture complete coverage',lambda:None if not audit.coverage_from_table(changed,[])[1] else (_ for _ in ()).throw(AssertionError()))
run('missing long timing remains partial',lambda:None if not audit.coverage_from_table(coverage_table[:5],[])[1] else (_ for _ in ()).throw(AssertionError()))

report={'passed':all(x['passed'] for x in tests),'gpu_used':False,
        'scope':'Synthetic saved tensor/timing/Adam fixtures only; no real large-batch inference or throughput result.',
        'auditor_sha256':audit.sha(HERE/'verify_large_batch_results.py'),
        'test_count':len(tests),'tests':tests}
audit.write(HERE/'large_batch_auditor_cpu_verification.json',report)
print(json.dumps({'passed':report['passed'],'test_count':len(tests)}))
