"""Produce NEW inactive TSC domain30 orchestration/routing drafts; stdlib only."""
import ast, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
routes=('cuda_cache_caller_tsc','launch_with_io_retry_tsc','launch_with_cache_reclaim_tsc',
        'launch_with_memory_guard_tsc','launch_native_training_tsc')
families=(*routes,'launch_with_memory_guard_TSC_resume','run_hf_TSC_arm',
          'resume_TSC_same_science','launch_TSC_same_science_resume','supervise_TSC_same_science')
names={stem+'_v2.py':stem+'_v3.py' for stem in families}
names['train_lora_batched_tsc_v2.py']='train_lora_batched_tsc_v3.py'

def change(text):
    for old,new in names.items():text=text.replace(old,new).replace(old[:-3],new[:-3])
    for old,new in (('isolated_HF_TSC_guarded_quality_arm_v2','isolated_HF_TSC_guarded_quality_arm_v3'),
        ('TSC_same_run_exact_science_continuation_v2','TSC_same_run_exact_science_continuation_v3'),
        ('TSC_same_science_native_preload_resume_v2','TSC_same_science_native_preload_resume_v3'),
        ('TSC_same_science_native_preload_resume_completion_v2','TSC_same_science_native_preload_resume_completion_v3'),
        ('memory_guard_training_launcher_TSC_resume_v2','memory_guard_training_launcher_TSC_resume_v3'),
        ('memory_guard_training_launcher_tsc_v2','memory_guard_training_launcher_tsc_v3'),
        ('cache_reclamation_training_launcher_tsc_v2','cache_reclamation_training_launcher_tsc_v3'),
        ('atomic_io_retry_tsc_v2','atomic_io_retry_tsc_v3'),
        ('isolated_TSC_runtime_chain_pins_v2','isolated_TSC_runtime_chain_pins_v3'),
        ('TSC_runtime_chain_pins_v2.json','TSC_runtime_chain_pins_v3.json'),
        ('pinned_TSC_controller_child_close_publication_source_proof_v2','pinned_TSC_controller_child_close_publication_source_proof_v3')):
        text=text.replace(old,new)
    return text

def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Expected one source span: '+old[:95])
    return text.replace(old,new)

for old,new in names.items():
    if old=='train_lora_batched_tsc_v2.py':continue
    text=change((HERE/old).read_text(encoding='utf-8'))
    if old.startswith('run_hf_TSC_arm'):
        text=once(text,'from tsc_training_inputs_v2 import closed_source_identity, approve, common_source_recipe as approved_common_source_recipe',
            'from tsc_training_inputs_v2 import approve\nfrom tsc_domain_training_identity_v3 import closed_source_identity, approved_source_recipe\nfrom tsc_domain_sampler_v3 import domain_horizon')
        text=once(text,"RECIPE_KEYS = ('epochs'", "RECIPE_KEYS = ('sampling_mode','fleurs_fraction','replay_seed','epochs'")
        text=once(text,"SOURCE_SAMPLER_KEYS = ('sampling_mode','fleurs_fraction','replay_seed')", "SOURCE_SAMPLER_KEYS = ('sampling_mode','fleurs_fraction','replay_seed')\nMINIMUM_DOMAIN_FLEURS_CLIPS = 2037")
        start=text.index('def common_source_recipe(');end=text.index('\n\ndef prepare(',start)
        text=text[:start]+'''def domain_counts(config, combined_count=None):
    """Only validated published pool metadata, never infer a uniform epoch."""
    coverage=config.get('sampling',{}).get('initial_pool',{})
    pools=coverage.get('domain',{})
    require(set(pools)=={'fleurs','cv_tsc'},'Published FLEURS/CV+TSC domain pools required')
    counts={key:pools[key]['pool_clips'] for key in ('fleurs','cv_tsc')}
    require(all(type(value) is int and value>0 for value in counts.values())
        and counts['fleurs']==MINIMUM_DOMAIN_FLEURS_CLIPS
        and sum(counts.values())==(combined_count if combined_count is not None else config['turkish_clips_used']),
        'Actual combined domain counts differ')
    return counts


def source_sampling_transition(source_recipe):
    require(set(source_recipe)==set(RECIPE_KEYS) and source_recipe['sampling_mode']=='domain30'
        and source_recipe['fleurs_fraction']==.3 and type(source_recipe['replay_seed']) is int,
        'Preserve explicit source30% FLEURS and independent replay science')
    return {'source_sampling_mode':'domain30','target_sampling_mode':'domain30',
        'source_fleurs_fraction':.3,'target_fleurs_fraction':.3,
        'source_replay_seed':source_recipe['replay_seed'],'target_replay_seed':source_recipe['replay_seed'],
        'source_and_target_replay_rng_independent':True,'replay_pool_teacher_targets_unchanged':True,
        'new_local_replay_stream_restarts_from_same_explicit_seed':True,
        'replay_sequence_matches_new_uniform_comparator_with_same_replay_seed':True,
        'replay_sequence_matches_parent_training_frontier':False,
        'other_TR_domain_change':'uniform CV expands to uniform CV+post-quarantine TSC',
        'fresh_Adam_sampler_RNG_local_counters':True}
''' +text[end:]
        text=once(text,"source_recipe=config['config_identity']['training_arguments'];recipe,sampler_transition=common_source_recipe(source_recipe)\n    require(approved_common_source_recipe(config,closure)==recipe,\n        'ROOT-approved CLOSED source sampler transition differs from the copied common science')\n    recipe.update(epochs=1.,steps=math.ceil(proof['clips']/8))", "source_recipe=config['config_identity']['training_arguments']\n    recipe=approved_source_recipe(config,closure);sampler_transition=source_sampling_transition(source_recipe)\n    require(recipe==source_recipe,'Copy all35 CLOSED domain recipe fields before the declared new data/horizon')\n    counts={'fleurs':MINIMUM_DOMAIN_FLEURS_CLIPS,'cv_tsc':proof['clips']-MINIMUM_DOMAIN_FLEURS_CLIPS}\n    recipe=recipe.copy();recipe.update(epochs=1.,steps=domain_horizon(counts['fleurs'],counts['cv_tsc']))")
        text=once(text,"'copied_common_source_recipe':{key:source_recipe[key] for key in RECIPE_KEYS},'source_sampler_transition':sampler_transition,", "'copied_common_source_recipe':{key:source_recipe[key] for key in RECIPE_KEYS if key not in SOURCE_SAMPLER_KEYS},'source_sampler_transition':sampler_transition,")
        text=once(text,"'configured_steps':recipe['steps'],'horizon':{'configured_updates':recipe['steps'],'uniform_combined_pool_epochs':1,\n            'quota_basis':'ceil(approved combined TR unique clips /8)','legacy_uniform_or_domain_cap_copied':False},", "'configured_steps':recipe['steps'],'domain_pool_sizes':counts,\n        'horizon':{'configured_updates':recipe['steps'],'minimum_domain_exposure_epochs':1,\n            'quota_basis':'First update with floor(TRdraws*3/10)>=FLEURS_N and remaining draws>=CV_TSC_N',\n            'legacy_uniform_or_domain_cap_copied':False,'full_pass_claim_requires_every_unique_source_seen':True},")
        text=once(text,"'One combined-pool pass; horizon calculated from actual approved clip count',\n            'Explicit source domain-weighting/independent-replay to combined-uniform/shared-replay sampler transition when applicable',\n            'Existing shared RNG retains changed-pool replay-order nuisance; unchanged replay order is not claimed'", "'One minimum-domain exposure horizon from actual approved domain counts, not one draw-equivalent epoch',\n            'Preserve30%FLEURS/70%other TR and independent replay seed; other domain expands CV to CV+TSC',\n            'Initial fresh sampler restarts local streams; parent replay frontier equality is not claimed'")
        start=text.index('def verify_pool_frontier(');end=text.index('\n\ndef verify_completion(',start)
        text=text[:start]+'''def verify_pool_frontier(config,selection,history,plan):
    count=plan['combined_TR_clips'];counts=domain_counts(config,count)
    require(plan['domain_pool_sizes']==counts,'Actual versus planned domains differ')
    horizon=domain_horizon(counts['fleurs'],counts['cv_tsc']);updates=selection['completed_steps']
    require(type(updates) is int and 0<=updates<=horizon and horizon==plan['configured_steps']
        and config.get('epochs')==1. and config.get('steps')==horizon
        and config.get('turkish_clips_used')==count and config.get('replay_clips_used')==1595
        and config.get('tr_clips_per_update')==8 and config.get('replay_clips_per_update')==8,
        'Actual minimum-domain horizon/effective16 geometry differs')
    require(len(history)==updates and [r['step'] for r in history]==list(range(1,updates+1)),
        'Actual ordered full history frontier differs')
    coverage=selection.get('Turkish_domain_coverage',{})
    require(coverage.get('epoch_clock')=='minimum_domain_exposure'
        and set(coverage.get('domain',{}))==set(counts), 'Actual domain coverage required')
    draws=updates*8;nf=draws*3//10;expected={'fleurs':nf,'cv_tsc':draws-nf}
    for key,size in counts.items():
        item=coverage['domain'][key]
        require(item['pool_clips']==size and item['draws']==expected[key]
            and item['unique_clips_seen']==min(expected[key],size)
            and item['repeat_visits']==expected[key]-min(expected[key],size),
            'Without-replacement domain unique/draw/repeat counters differ: '+key)
    epoch=min(expected[key]/size for key,size in counts.items())
    unique=sum(min(expected[key],size) for key,size in counts.items())
    require(selection['turkish_epochs_completed']==epoch
        and selection.get('turkish_epoch_semantics')=='minimum_domain_exposure'
        and coverage.get('minimum_domain_exposure_epochs')==epoch
        and coverage.get('turkish_draws')==draws and coverage.get('turkish_unique_clips_seen')==unique,
        'Minimum-domain clock must remain distinct from draw-equivalent global coverage')
    if history:
        require(history[-1]['turkish_clips_seen']==draws and history[-1]['turkish_unique_clips_seen']==unique
            and history[-1].get('Turkish_domain_coverage')==coverage,'Actual final history coverage differs')
    if updates==horizon:
        require(epoch>=1. and coverage.get('all_Turkish_pool_sources_seen') is True,
            'Claiming full minimum-domain exposure requires all unique sources seen')
    return {'combined_TR_clips':count,'domain_pool_sizes':counts,'configured_steps':horizon,
        'TR_draws':draws,'unique_TR_seen':unique,'minimum_domain_exposure_epochs_completed':epoch,
        'turkish_draw_equivalent_epochs':draws/count,'actual_domain_coverage':coverage}
''' +text[end:]
        text=text.replace('uniform_combined_pool_epochs_completed','minimum_domain_exposure_epochs_completed')
        text=text.replace('configured_combined_pool_pass_completed','configured_minimum_domain_pass_completed')
        text=text.replace('uniform combined-pool coverage','minimum-domain and unique-source coverage')
        text=text.replace('data-only arm','domain-preserving TSC data arm')
        text=text.replace('One combined-pool pass','One minimum-domain exposure pass')
        ds=text.index('"""');de=text.index('"""',ds+3)
        text='''"""Guarded first approved TSC TRAIN domain-preserving arm; no TEST/export.

Keep the CLOSED source's30%FLEURS/70%otherTR and independent replay RNG.
Only expand the other domain from CV to uniform CV+approved post-quarantine TSC.
Fresh Adam initially; scheduled windows later preserve SAME-run full state.
The minimum-domain horizon is not a promise to continue past a ROOT quality decision.
"""'''+text[de+3:]
    elif old.startswith('resume_TSC_same_science'):
        text=once(text,"count=config['turkish_clips_used'];horizon=math.ceil(count/8)","count=config['turkish_clips_used'];counts=domain.domain_counts(config,count)\n    horizon=domain.domain_horizon(counts['fleurs'],counts['cv_tsc'])")
        text=once(text,"'Keep one uniform combined-pool pass and original ceil(N/8) frontier'", "'Keep original one minimum-domain exposure horizon and full domain sampler frontier'")
        text=once(text,"original_common,source_sampler_transition=domain.common_source_recipe(source_recipe)\n    require(domain.approved_common_source_recipe(original_source_config,closure)==original_common,\n        'Original ROOT-approved source sampler transition changed')", "original_common=domain.approved_source_recipe(original_source_config,closure)\n    domain.source_sampling_transition(source_recipe)\n    require(original_common==source_recipe,'Original ROOT-approved35-field domain science changed')")
        text=once(text,"'combined_TR_clips':config['turkish_clips_used'],'configured_steps':config['steps'],", "'combined_TR_clips':config['turkish_clips_used'],'configured_steps':config['steps'],\n        'domain_pool_sizes':domain.domain_counts(config),")
    elif old.startswith('supervise_TSC_same_science'):
        text=once(text,"require(config.get('steps')==math.ceil(config['turkish_clips_used']/8) and identity['training_arguments']['epochs']==1.,\n        'Preserve actual one combined-pool pass ceil(N/8) horizon')", "counts=domain.domain_counts(config)\n    require(config.get('steps')==domain.domain_horizon(counts['fleurs'],counts['cv_tsc'])\n        and identity['training_arguments']['epochs']==1.,\n        'Preserve actual one minimum-domain exposure horizon and fullstate frontier')")
        text=text.replace('one combined-pool pass','one minimum-domain exposure pass')
        text=text.replace('unchanged original v2 TSC science chain','unchanged original v3 domain-preserving TSC science chain')
    # Concrete hashes are wired only after retention freezes its new science sources.
    if old.startswith(('run_hf_TSC_arm','resume_TSC_same_science','launch_TSC_same_science_resume','supervise_TSC_same_science')):
        text=re.sub(r"^(CHAIN_SHA|TSC_CONTROLLER_SHA|FIRST_SHA|HELPER_SHA|PRELOADER_SHA|RUNTIME_MEMORY_SHA)\s*=.*$",lambda m:m.group(1)+" = 'PENDING_FINAL_SOURCE'",text,flags=re.M)
    if old.startswith('launch_TSC_same_science_resume'):
        text=re.sub(r'^PINS = .*$',"PINS = {}  # Pending actual new science/runtime chain",text,flags=re.M)
    ast.parse(text)
    target=HERE/new
    if target.exists():raise FileExistsError('Drafts are created once; edit new files explicitly: '+str(target))
    target.write_text(text,encoding='utf-8')
print(json.dumps({'drafts':list(names.values())[:-1],'ML_imported':False,'GPU_launched':False}))
