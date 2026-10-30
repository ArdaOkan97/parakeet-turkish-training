"""Small stdlib guards and TRAIN-only domain/coverage measurements; no Torch."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import turkish_domain_sampler_v1 as module
from atomic_io_retry_v1 import read_json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
worker=OUT/'train_lora_batched_data_v11.py';tree=ast.parse(worker.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if getattr(n,'name',None) in ('BucketCycler','ids_sha')]
scope={'math':math,'hashlib':hashlib}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(worker),'exec'),scope)
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def trrow(domain,i):
    return {'source_id':domain+str(i),'language':'tr','encoder_frames':7+i,'target_tokens':3+i%9,'duration':1+i/10,
        'dataset':'google/fleurs' if domain=='fleurs' else 'ysdede/commonvoice_17_tr_fixed',
        'dataset_revision':module.FLEURS_REVISION if domain=='fleurs' else module.CV_REVISION,
        'dataset_config':'tr_tr' if domain=='fleurs' else 'default','dataset_split':'train'}
tr=[trrow('fleurs',i) for i in range(17)]+[trrow('cv',i) for i in range(53)]
replay=[{'source_id':lang+str(i),'language':lang,'encoder_frames':12+i,'target_tokens':8+i,'duration':2+i/10}
    for lang in module.LANGUAGES for i in range(7)]
def sampler(**overrides):
    kwargs={'seed':20261010,'batch_size':8,'accumulation':2,'replay_fraction':.5,'fleurs_fraction':.3,
        'bucket_size':8,'cycler_factory':scope['BucketCycler'],'cycler_source_sha256':sha(worker)}
    kwargs.update(overrides)
    return module.DomainBalancedBucketSampler(tr,replay,**kwargs)
global_rng=random.getstate();s=sampler()
check('no_global_Python_RNG_change',random.getstate()==global_rng)
for i in range(100):
    batches=s.next_batches();check('effective16_balanced_TR8_replay8_update_'+str(i),len(batches)==2
        and all(len(b)==8 for b in batches) and sum(r['language']=='tr' for b in batches for r in b)==8)
state=s.state_dict()
check('exact_30percent_clip_quota_after800TR',state['domains']['fleurs']['draws']==240 and state['domains']['cv']['draws']==560)
check('unchanged50percent_replay_quota',sum(v['draws'] for v in state['replay'].values())==800)
check('every_domain_source_seen',s.coverage()['all_Turkish_pool_sources_seen'])
check('repeat_visits_reported',s.coverage()['domain']['fleurs']['repeat_visits']==223)
s=sampler()
for _ in range(33):s.next_batches()
saved=pickle.loads(pickle.dumps(s.state_dict(),protocol=5));restored=sampler();restored.load_state_dict(saved)
check('checkpoint_pickle_preserves_RNG_and_cursors',restored.state_dict()==saved)
for i in range(20):
    check('exact_resume_next_update_'+str(i),[[r['source_id'] for r in b] for b in s.next_batches()]
        ==[[r['source_id'] for r in b] for b in restored.next_batches()])
for field in ('updates','rng'):
    check('resume_state_'+field+'_matches',s.state_dict()[field]==restored.state_dict()[field])
bad=copy.deepcopy(saved);bad['domains']['fleurs']['draws']+=1
try:sampler().load_state_dict(bad)
except ValueError:checks.append('tampered_domain_quota_rejected')
else:raise AssertionError('Tampered draws accepted')
try:sampler(fleurs_fraction=.4).load_state_dict(saved)
except ValueError:checks.append('changed_domain_recipe_rejected_on_exact_resume')
else:raise AssertionError('Changed ratio accepted')
bad=dict(tr[0],dataset_split='validation')
try:module.turkish_domain(bad)
except ValueError:checks.append('heldout_split_never_domain_eligible')
else:raise AssertionError('Heldout accepted')
s=sampler()
for _ in range(9):s.next_batches()
coverage=s.coverage()
check('nominal_draw_epoch_not_false_full_pass',coverage['turkish_draw_equivalent_epochs']>1
    and not coverage['all_Turkish_pool_sources_seen'] and coverage['minimum_domain_exposure_epochs']<1)
check('coverage_corrected_onepass_budget',s.updates_for_minimum_domain_exposure(1)==10)

# Compact immutable TRAIN-only metadata, no waveform/model or heldout references.
totals={name:{'clips':0,'hours':0.} for name in ('fleurs','cv')};ids={}
manifest=ROOT/'work/data/tr_public_v3_train.jsonl'
with manifest.open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line);domain=module.turkish_domain(row)
        totals[domain]['clips']+=1;totals[domain]['hours']+=row['duration']/3600
        ids[row['source_id']]=domain
N=sum(v['clips'] for v in totals.values())
for value in totals.values():value['mean_duration_seconds']=value['hours']*3600/value['clips']
f,c=totals['fleurs'],totals['cv']
nominal_draws=N
nominal_f=math.floor(.3*nominal_draws);nominal_c=nominal_draws-nominal_f
full_draws=math.ceil(c['clips']/.7)
while full_draws-math.floor(.3*full_draws)<c['clips']:full_draws+=1
old_path=OUT/'runs/adaptive_20261011_hf_return_01/training_history.json'
new_path=OUT/'runs/adaptive_20261011_hf_quality_window_01_stronger_kd/training_history.json'
old=read_json(old_path,[]);new=read_json(new_path,[])
old_TR=[sid for row in old for sid in row['sampled_source_ids'] if sid in ids]
new_TR=[sid for row in new for sid in row['sampled_source_ids'] if sid in ids]
coverage_actual={'old_closed_updates':len(old),'old_TR_draws':len(old_TR),'old_TR_unique':len(set(old_TR)),
    'old_TR_unique_fraction':len(set(old_TR))/N,'fresh_arm_published_updates':len(new),
    'fresh_arm_TR_draws':len(new_TR),'fresh_arm_TR_unique':len(set(new_TR)),
    'fresh_arm_old_prefix_sequence_equal':new_TR==old_TR[:len(new_TR)],
    'fresh_arm_TR_unique_overlap_with_closed':len(set(new_TR)&set(old_TR)),
    'fresh_arm_new_distinct_TR_over_closed':len(set(new_TR)-set(old_TR)),
    'observed_fresh_prefix_only_not_future_claim':True}
report={'schema':1,'kind':'Turkish_domain_sampler_preparation_v1','passed':True,'status':'inert_CPU_prepared_not_integrated',
    'check_count':len(checks),'checks':checks,'module_sha256':sha(Path(module.__file__)),'cycler_source_sha256':sha(worker),
    'manifest':{'path':str(manifest),'sha256':sha(manifest)},'pools':totals,'pool_total':N,
    'proposed_clip_fraction':{'fleurs':.3,'cv':.7},'replay_fraction':.5,
    'FLEURS_clip_oversampling_multiplier':.3/(f['clips']/N),
    'expected_FLEURS_share_of_Turkish_audio_from_mean_durations':.3*f['mean_duration_seconds']/(.3*f['mean_duration_seconds']+.7*c['mean_duration_seconds']),
    'nominal_N_TR_draws':{'draws':N,'fleurs_draws':nominal_f,'cv_draws':nominal_c,
        'maximum_cv_unique_fraction':nominal_c/c['clips'],'full_pool_pass':False},
    'minimum_total_TR_draws_for_one_CV_cycle':full_draws,
    'B8x2_TR8_updates_for_one_domain_coverage_pass':math.ceil(full_draws/8),
    'fresh_arm_repeated_prefix':coverage_actual,'source_history':{'path':str(old_path),'sha256':sha(old_path)},
    'live_history_snapshot':{'path':str(new_path),'sha256_at_read':sha(new_path) if new_path.exists() else None},
    'actual_Torch_or_NeMo_imported':False,'GPU_launch':False,
    'integration_requirements':['New declared sampling recipe/checkpoint identity; never load an old uniform sampler state as domain state',
        'Replace old scalar epoch/step-budget assumptions with domain coverage metrics and updates_for_minimum_domain_exposure',
        'Retain accepted source/label/audio metadata and fixed heldout isolation proof',
        'Measure actual padding/throughput: mixed FLEURS/CV lengths can raise B8 lattice padding despite per-domain reservoirs']}
path=OUT/'turkish_domain_sampler_preparation_v1.json'
with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
print(json.dumps({'passed':True,'checks':len(checks),'module_sha256':report['module_sha256'],'pools':totals,
    'coverage_actual':coverage_actual,'one_domain_pass_updates':report['B8x2_TR8_updates_for_one_domain_coverage_pass']},indent=2))
