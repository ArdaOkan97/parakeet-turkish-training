"""NEW inactive logger derivative; this constructor never imports Trackio."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v11.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='6a58eb0e8ccf25dae1263afcfc356f1e361d832d4d252698593edcb83d850b1e'
t=source.read_text(encoding='utf-8')
t=t.replace('INACTIVE Trackio mirror v11: full uniform Turkish pass with Portuguese.','INACTIVE Trackio mirror v12: declared same-pass14TR/2replay continuation.')
t=t.replace('V8/V9/V10 archives are preserved. ROOT must confirm V10 CLOSED before handoff.','V8/V9/V10/V11 archives are preserved. ROOT must confirm V11 CLOSED before handoff.')
t=t.replace("SOURCE_V10_SHA='935309ae110cb37b8c690a6389dc260bc3aded83eab053652712459d898b4f54'", "SOURCE_V11_SHA='6a58eb0e8ccf25dae1263afcfc356f1e361d832d4d252698593edcb83d850b1e'")
t=t.replace("UNIFORM_FULL_PHASE='HF_TSC_8199_uniform_fullpass_with_Portuguese_v1'", "UNIFORM_FULL_PHASE='HF_TSC_8199_uniform_fullpass_with_Portuguese_v1'\nUNIFORM_MAJORITY_PHASE='HF_TSC_8199_uniform_Turkish_majority_with_Portuguese_v1'")
new='''
def uniform_majority_context(config,history):
    if config.get('phase')!=UNIFORM_MAJORITY_PHASE:return None
    h=config.get('source_step');origin=config.get('uniform_origin_step');full=config.get('uniform_full_pass',{})
    identity=config.get('config_identity',{});recipe=identity.get('training_arguments',{})
    declared=identity.get('majority_transition',{}).get('declared',{});proof=config.get('initial_source_equivalence',{})
    prior=full.get('prior_uniform_draws');pool=full.get('expected_new_unique_TR_clips');budget=full.get('additional_update_budget')
    if (type(h) is not int or h<=6500 or origin!=6500 or pool!=253798 or prior!=8*(h-origin)
        or not 0<prior<pool or budget!=math.ceil((pool-prior)/14) or config.get('steps')!=h+budget
        or full.get('source_step')!=origin or full.get('transition_source_step')!=h or full.get('end_step')!=h+budget
        or config.get('inherited_history_steps')!=h or config.get('trainable_parameter_tensors')!=207
        or config.get('replay_clips_used')!=1914 or declared.get('source_step')!=h
        or declared.get('prior_uniform_draws')!=prior or declared.get('TR_per_update')!=14 or declared.get('replay_per_update')!=2
        or declared.get('configured_end_step')!=h+budget or declared.get('renewed_Portuguese_grace') is not False
        or declared.get('preservation_coefficients')!={'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.}
        or recipe.get('replay_fraction')!=.125 or recipe.get('replay_kl')!=40 or recipe.get('duration_kl')!=2 or recipe.get('feature_distillation')!=4
        or recipe.get('head_lr')!=1e-5 or recipe.get('lora_lr')!=3e-4 or recipe.get('new_token_lr_multiplier')!=6
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True or proof.get('source_selected_step')!=h
        or proof.get('before_first_majority_update') is not True or proof.get('compared_fixed_DEV_utterances')!=350
        or proof.get('all_seven_ASR_counts_exact') is not True):
        raise ValueError('Declared original207/sourceH same-pass14/2 and actual source350 identity required')
    base=next((row for row in history if row.get('step')==h),None)
    original=next((row for row in history if row.get('step')==origin),None)
    if base is None or original is None:raise ValueError('SourceH rates and original6500 coverage baselines required')
    local=base.get('Turkish_domain_coverage',{}).get('uniform_pass',{})
    if local.get('draws')!=prior or local.get('unique_clips_seen')!=prior or local.get('source_step')!=origin:
        raise ValueError('Actual closed source queue frontier differs')
    context={'step':h,'origin':origin,'prior':prior,'pool':pool,'budget':budget,'head_lr':recipe['head_lr'],'lora_lr':recipe['lora_lr'],
        'historical_unique':original.get('turkish_unique_clips_seen'),'source_TR_draws':base.get('turkish_clips_seen'),
        'source_RP_draws':base.get('replay_clips_seen'),'source_gates':proof['source_retention_gates'],
        'source_all_six_passed':proof['all_six_retention_gates_passed']}
    if type(context['historical_unique']) is not int or type(context['source_TR_draws']) is not int or type(context['source_RP_draws']) is not int:
        raise ValueError('Actual historical union/source draw counters required')
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        if not finite_number(base.get(key)):raise ValueError('Finite actual H counter required: '+key)
        context[key]=base[key]
    return context


def uniform_majority_values_for(record,context):
    if context is None or record['step']<context['step']:return {}
    n=record['step']-context['step'];full_n=record['step']-context['origin'];pool=context['pool']
    if not 0<=n<=context['budget']:raise ValueError('Majority localM beyond same-pass remaining bound')
    values={'phase/majority_updates':n,'phase/majority_source_step':context['step'],'phase/uniform_origin_step':context['origin'],
        'phase/uniform_full_updates':full_n,'phase/expected_full207_Adam_clock':record['step'],'phase/active_Adam_parameter_tensors':207,
        'phase/head_lr':context['head_lr'],'phase/lora_lr':context['lora_lr'],'phase/declared_TR_per_update':14,'phase/declared_RP_per_update':2,
        'phase/declared_replay_kl_coefficient':40.,'phase/declared_duration_kl_coefficient':2.,'phase/declared_feature_distillation_coefficient':4.,
        'phase/expected_RP_replay_KL_mean_density':5.,'phase/expected_RP_duration_KL_mean_density':.25,'phase/expected_RP_feature_mean_density':.5,
        'data/uniform_full_TR_pool_clips':pool,'data/historical_TR_unique_clips_at_6500':context['historical_unique'],
        'data/remaining_uniform_updates_at_H':context['budget'],
        'retention/Portuguese_recovery_grace_remaining_updates':max(0,1000-full_n)}
    if n==0:
        values.update({'data/uniform_local_TR_draws':context['prior'],'data/uniform_local_unique_TR_clips':context['prior'],
            'data/uniform_local_fraction_unique_seen':context['prior']/pool,'data/uniform_local_epoch':context['prior']/pool,
            'data/uniform_local_repeat_visits':0,'data/uniform_full_pass_complete':0})
        return values
    local=record.get('Turkish_domain_coverage',{}).get('uniform_pass',{})
    transition=record.get('Turkish_domain_coverage',{}).get('majority_transition',{})
    draws=context['prior']+14*n;unique=min(draws,pool);union=record.get('turkish_unique_clips_seen')
    expected_segments=[{'first_step':6501,'last_step':context['step'],'updates':context['step']-6500,'TR_per_update':8,'replay_per_update':8,
        'TR_draws':context['prior'],'replay_draws':context['prior']},
        {'first_step':context['step']+1,'last_step':record['step'],'updates':n,'TR_per_update':14,'replay_per_update':2,'TR_draws':14*n,'replay_draws':2*n}]
    if (record.get('majority_updates')!=n or record.get('majority_source_step')!=context['step'] or record.get('uniform_origin_step')!=6500
        or record.get('uniform_full_updates')!=full_n or record.get('head_lr')!=context['head_lr'] or record.get('lora_lr')!=context['lora_lr']
        or record.get('new_token_update_multiplier')!=6 or record.get('replay_kl_coefficient')!=40
        or record.get('duration_kl_coefficient')!=2 or record.get('feature_distillation_coefficient')!=4
        or local.get('source_step')!=6500 or local.get('local_updates')!=full_n or local.get('pool_clips')!=pool
        or local.get('draws')!=draws or local.get('unique_clips_seen')!=unique or local.get('repeat_visits')!=draws-unique
        or local.get('fraction_unique_seen')!=unique/pool or local.get('epoch_clock')!=draws/pool
        or local.get('full_pass_complete')!=(unique==pool) or local.get('quota_segments')!=expected_segments
        or transition.get('local_updates')!=n or transition.get('source_step')!=context['step']
        or record.get('turkish_clips_seen')!=context['source_TR_draws']+14*n or record.get('replay_clips_seen')!=context['source_RP_draws']+2*n
        or type(union) is not int or not max(unique,context['historical_unique'])<=union<=min(pool,unique+context['historical_unique'])):
        raise ValueError('Actual14/2 piecewise coverage/LR/density/global counters disagree')
    delta={}
    for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds'):
        if not finite_number(record.get(key)):raise ValueError('Finite majority counter required')
        delta[key]=record[key]-context[key]
        if delta[key]<=0:raise ValueError('PostH mixed/TR/replay/complete-update counters must increase')
    if not math.isclose(delta['audio_seen_seconds'],delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('PostH audio is not TR plus replay')
    wall=delta['train_wall_seconds']
    values.update({'data/uniform_local_TR_draws':draws,'data/uniform_local_unique_TR_clips':unique,
        'data/uniform_local_fraction_unique_seen':unique/pool,'data/uniform_local_epoch':draws/pool,'data/uniform_local_repeat_visits':draws-unique,
        'data/uniform_full_pass_complete':int(unique==pool),'data/cumulative_TR_unique_union':union,
        'data/majority_TR_draws':14*n,'data/majority_RP_draws':2*n,
        'resources/post_majority_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_majority_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_majority_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_majority_train_wall_seconds':wall,
        'data/post_majority_audio_seen_seconds':delta['audio_seen_seconds'],
        'data/post_majority_turkish_audio_seen_seconds':delta['turkish_audio_seen_seconds'],
        'data/post_majority_replay_audio_seen_seconds':delta['replay_audio_seen_seconds']})
    gate=record.get('validation')
    if gate is not None:
        if gate.get('Portuguese_grace_active_for_stop_only')!=(full_n<1000):raise ValueError('Portuguese grace must remain anchored6500')
        values.update({'retention/all_six_passed':int(gate['retention_gate_passed']),
            'retention/original_five_passed':int(gate['old_five_retention_gate_passed']),
            'retention/Portuguese_stop_grace_active':int(gate['Portuguese_grace_active_for_stop_only']),
            'retention/stop_guard_passed':int(gate['stop_retention_gate_passed'])})
    return values


def majority_phase_baseline_steps(config):
    if config.get('phase')==UNIFORM_MAJORITY_PHASE:return {6500,config['source_step']}
    return full_phase_baseline_steps(config)


def majority_phase_initial_development_step(config):
    return config['source_step'] if config.get('phase')==UNIFORM_MAJORITY_PHASE else full_phase_initial_development_step(config)


def majority_phase_actual_development_row(config,record):
    if config.get('phase')==UNIFORM_MAJORITY_PHASE:
        return 'validation' in record and record['step']>config['source_step'] and record.get('majority_updates')==record['step']-config['source_step']
    return full_phase_actual_development_row(config,record)


def majority_phase_contexts_for(config,baseline_rows):
    contexts=full_phase_contexts_for(config,baseline_rows)
    contexts['majority']=uniform_majority_context(config,baseline_rows)
    return contexts


def majority_phase_value_for_record(record,contexts,development):
    values=full_phase_value_for_record(record,contexts,development);context=contexts.get('majority')
    if context is None:return values
    if record['step']<=context['step']:
        values={key:value for key,value in values.items() if not key.startswith(('retention/','dev/'))}
        values['phase/inherited_history']=int(record['step']<context['step'])
        actual=development.get(record['step'],{}) if record['step']==context['step'] else {}
        if set(actual)=={'dev/wer_'+language+'_percent' for language in ('tr','en','es','it','fr','de','pt')}:
            values.update(actual);values['retention/all_dev_gates_passed']=int(context['source_all_six_passed'])
            values['retention/all_six_passed']=int(context['source_all_six_passed'])
            values['retention/original_five_passed']=int(all(context['source_gates'][language]['passed'] for language in ('en','es','it','fr','de')))
            values['retention/Portuguese_stop_grace_active']=int(context['step']-6500<1000)
    values.update(uniform_majority_values_for(record,context));return values


def majority_selection_values(config,selected,last_step):
    if config.get('phase')!=UNIFORM_MAJORITY_PHASE:return uniform_full_selection_values(config,selected,last_step)
    if selected.get('run_id')!=config.get('run_id'):return {}
    h=config['source_step'];step=selected.get('completed_steps');n=selected.get('majority_updates');chosen=selected.get('selected_step')
    if (type(n) is not int or not 0<=n<=config['remaining_uniform_pass_updates'] or step!=h+n or step>last_step
        or selected.get('uniform_origin_step')!=6500 or selected.get('uniform_full_updates')!=step-6500
        or type(selected.get('accepted')) is not bool or type(selected.get('full_new_uniform_pass_complete')) is not bool
        or selected['full_new_uniform_pass_complete']!=(n==config['remaining_uniform_pass_updates'])):
        raise ValueError('Actual majority selection frontier differs')
    result={'selection/all_six_eligible_best_available':int(selected['accepted']),'selection/published_majority_updates':n,
        'selection/published_uniform_updates':step-6500,'selection/published_full_pass_complete':int(selected['full_new_uniform_pass_complete'])}
    if selected['accepted']:
        if type(chosen) is not int or not 6500<chosen<=step or not finite_number(selected.get('turkish_validation_wer')):raise ValueError('Eligible all6 selected checkpoint required')
        result.update({'selection/best_step':chosen,'selection/best_qualifying_TR_wer_percent':100*selected['turkish_validation_wer'],
            'selection/eligible_BEST_inherited_at_or_before_H':int(chosen<=h)})
    elif chosen is not None or selected.get('turkish_validation_wer') is not None:raise ValueError('No eligible all6 BEST must remain None')
    return result


def majority_rejected_run_terminal(config,selected):
    if config.get('phase')!=UNIFORM_MAJORITY_PHASE:return rejected_run_terminal(config,selected)
    return selected.get('full_new_uniform_pass_complete') is True or selected.get('stop_early_reason') in ('consecutive_retention_gate_failures','development_plateau','development_target_met')

'''
marker='\ndef file_record(path):'
t=t.replace(marker,'\n'+new+marker,1)
# Redirect the orchestration only. Every existing phase/math function remains
# byte-for-byte in source and AST; new wrappers delegate unchanged legacy work.
tree=ast.parse(t)
for name in ('HistoryScanCache','follow'):
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    lines=t.splitlines(keepends=True);block=''.join(lines[node.lineno-1:node.end_lineno])
    for old,newname in [('full_phase_baseline_steps','majority_phase_baseline_steps'),('full_phase_initial_development_step','majority_phase_initial_development_step'),
        ('full_phase_actual_development_row','majority_phase_actual_development_row'),('full_phase_contexts_for','majority_phase_contexts_for'),
        ('full_phase_value_for_record','majority_phase_value_for_record'),('uniform_full_selection_values','majority_selection_values'),
        ('rejected_run_terminal','majority_rejected_run_terminal')]:block=block.replace(old,newname)
    t=''.join(lines[:node.lineno-1])+block+''.join(lines[node.end_lineno:]);tree=ast.parse(t)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamRunShardStore')
lines=t.splitlines(keepends=True);block=''.join(lines[node.lineno-1:node.end_lineno])
for old,newname in [('Trackio_run_shard_index_v11','Trackio_run_shard_index_v12'),('Trackio_run_shard_index_v10','Trackio_run_shard_index_v11'),
    ('track_continuation_v10.py','track_continuation_v11.py'),('SOURCE_V10_SHA','SOURCE_V11_SHA'),('source_v10_script','source_v11_script'),
    ('source_V10_index_payload','source_V11_index_payload'),('V10 source/V11 index','V11 source/V12 index'),('V10 CLOSED before V11','V11 CLOSED before V12'),
    ('Frozen V10','Frozen V11'),('Complete pinned V10','Complete pinned V11'),('V10 index changed','V11 index changed'),
    ('Invalid/incomplete V11 index','Invalid/incomplete V12 index'),('source_V8_V9_V10_archive_pruned','source_V8_V9_V10_V11_archive_pruned')]:block=block.replace(old,newname)
t=''.join(lines[:node.lineno-1])+block+''.join(lines[node.end_lineno:])
t=t.replace("default=ROOT/'work/trackio_continuation_mirror_state_v11.json'", "default=ROOT/'work/trackio_continuation_mirror_state_v12.json'")
t=t.replace("default=ROOT/'work/trackio_continuation_mirror_state_v10.json'", "default=ROOT/'work/trackio_continuation_mirror_state_v11.json'")
t=t.replace('ROOT observed source V10 process CLOSED','ROOT observed source V11 process CLOSED')
ast.parse(t)
destination=HERE/'track_continuation_v12.py'
if destination.exists():raise ValueError('NEW V12 already exists')
destination.write_text(t,encoding='utf-8',newline='\n')
print(hashlib.sha256(destination.read_bytes()).hexdigest())
