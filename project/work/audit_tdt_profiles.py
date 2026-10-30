"""Read-only CPU evidence audit of recorded actual CUDA profiling."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
profiles={name:json.loads((OUTPUT/f'tdt_profile_{name}.json').read_text(encoding='utf-8')) for name in ('full','reduced','lora','parity')}
checks=[]

def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})
    if not condition:
        raise AssertionError(name)

def finite(value):
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)

selection=hashlib.sha256((OUTPUT/'memory_profile_examples.json').read_bytes()).hexdigest()
base=profiles['full']
measurements={}
for name,result in profiles.items():
    check('status '+name,result['status']==('passed' if name=='parity' else 'complete'))
    for key in ('seed','chunk_size','sigma','replay_kl_coefficient','examples','base_path','hardware','vocabulary_extension'):
        check('identical input/model configuration '+name+key,result[key]==base[key])
    check('selection byte identity '+name,result['selection_sha256']==selection)
    for filename,digest in result['code_sha256'].items():
        check('actual recorded code SHA '+name+filename,hashlib.sha256((OUTPUT/filename).read_bytes()).hexdigest()==digest)
    check('all saved numeric fields finite '+name,finite(result))
    measure=result['measurement']
    if name=='parity':
        continue
    check('one accumulated Adam step '+name,measure['optimizer_steps']==1 and measure['microbatches']==2)
    check('identical sampled streams '+name,[v['stream'] for v in measure['objectives']]==['tr','replay'])
    gradient=measure['gradient_audit']
    check('finite present and frozen-free gradients '+name,gradient['all_trainable_gradients_finite_and_present'] and
          not gradient['missing_gradient_names'] and not gradient['nonfinite_gradient_names'] and not gradient['frozen_parameter_gradient_names'])
    memory=measure['memory_after_step']
    check('valid allocator peaks '+name,memory['peak_allocated_bytes']>=memory['allocated_bytes'] and
          memory['peak_reserved_bytes']>=memory['reserved_bytes']>=memory['allocated_bytes'])
    check('positive synchronized step time '+name,measure['seconds_one_accumulated_adam_step']>0)
    for objective in measure['objectives']:
        weighted=objective['tdt_loss']+result['replay_kl_coefficient']*objective['replay_kl']
        check('objective weighting '+name+objective['stream'],abs(weighted-objective['weighted_total_before_accumulation_division'])<2e-6)
    measurements[name]={'peak_allocated_GB':memory['peak_allocated_bytes']/1e9,
        'peak_reserved_GB':memory['peak_reserved_bytes']/1e9,
        'synchronized_seconds_one_accumulated_adam_step':measure['seconds_one_accumulated_adam_step'],
        'trainable_parameters':gradient['trainable_parameters'],'objectives':measure['objectives']}
    if name=='lora':
        metadata=result['encoder_lora']
        check('all24 rank8 projections',metadata['adapted_layer_indices']==list(range(24)) and len(metadata['targets'])==96 and
            all(v['rank']==8 and v['alpha']==16 and v['dropout']==0 for v in metadata['targets']))
        check('base encoder frozen',metadata['non_adapter_encoder_trainable_parameter_count']==0)
        check('all96 LoRA B receive gradients',gradient['adapter_nonzero_gradient_tensors']['lora_B']==96)
        check('zero-initialized LoRA A gradients expected',gradient['adapter_nonzero_gradient_tensors']['lora_A']==0)

parity=profiles['parity']['measurement']
check('actual short same-clip loss criterion',parity['weighted_loss_absolute_difference']<1e-4 and parity['tdt_loss_absolute_difference']<1e-4)
for mode in ('full','reduced'):
    check('parity finite gradients '+mode,parity['objective_summaries'][mode]['gradient_audit']['all_trainable_gradients_finite_and_present'])
    check('parity recorded weighted loss difference',abs(abs(parity['objective_summaries']['full']['weighted_loss']-
        parity['objective_summaries']['reduced']['weighted_loss'])-parity['weighted_loss_absolute_difference'])<1e-12)
details=parity['per_parameter']
check('all original head trainable tensors compared',len(details)==profiles['full']['measurement']['gradient_audit']['trainable_tensors'])
aggregate=(sum(v['difference_l2']**2 for v in details)/sum(v['reference_gradient_l2']**2 for v in details))**.5
check('aggregate head-gradient error recomputed',abs(aggregate-parity['all_head_gradient_relative_l2_difference'])<1e-12)
check('all head-gradient acceptance criteria',aggregate<.02 and parity['all_head_gradient_cosine']>.999 and
    all(v['relative_gradient_l2']<.02 and v['cosine_similarity']>.999 for v in details if v['reference_gradient_l2']>1e-8))
check('full vs reduced same long-clip objectives',all(abs(a['tdt_loss']-b['tdt_loss'])<1e-4 and abs(a['replay_kl']-b['replay_kl'])<1e-4
    for a,b in zip(measurements['full']['objectives'],measurements['reduced']['objectives'])))
audit={'passed':True,'gpu_used_by_audit':False,'checks':checks,'check_count':len(checks),
    'sources_sha256':{name:hashlib.sha256((OUTPUT/f'tdt_profile_{name}.json').read_bytes()).hexdigest() for name in profiles},
    'actual_measurements':measurements,
    'reduced_peak_allocated_reduction_percent':100*(1-measurements['reduced']['peak_allocated_GB']/measurements['full']['peak_allocated_GB']),
    'lora_vs_full_peak_allocated_reduction_percent':100*(1-measurements['lora']['peak_allocated_GB']/measurements['full']['peak_allocated_GB']),
    'gpu_head_parity':{key:parity[key] for key in ('weighted_loss_absolute_difference','tdt_loss_absolute_difference',
        'all_head_gradient_relative_l2_difference','all_head_gradient_cosine')},
    'limitations':['Single synchronized accumulated Adam step per mode, not replicated steady-state throughput.',
        'Same two cached stress recordings; full/reduced use frozen cached encoder outputs, LoRA processes their raw audio.',
        'Torch peaks measure allocator memory in the process, including weights/teacher/gradients/new Adam states; driver and other processes are excluded.',
        'Actual GPU parity compares all trainable decoder/projector/joint gradients on one short recording, not all-layer LoRA gradients against a full raw-audio reference.',
        'Finite LoRA gradients and 96 nonzero B gradients establish the training path, not recognition accuracy or preservation quality.']}
(OUTPUT/'tdt_profile_independent_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({key:audit[key] for key in ('passed','check_count','actual_measurements','reduced_peak_allocated_reduction_percent','gpu_head_parity')},indent=2))
