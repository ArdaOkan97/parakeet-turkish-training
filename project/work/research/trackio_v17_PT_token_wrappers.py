def PT_token_context(config,history):
    transition=config.get('PT_token_retention_constraint')
    if transition is None:return None
    identity=config.get('config_identity',{});proof=config.get('initial_source_equivalence',{});declared=transition.get('declared',{});h=config.get('source_step')
    if (config.get('phase')!=PT_FOCUSED_PHASE or h!=8750 or config.get('PT_focus_source_step')!=8000
        or identity.get('PT_token_retention_constraint')!=transition
        or transition.get('kind')!='TSC_uniform_PT_token_retention_constraint_full_state_v1' or transition.get('source_step')!=h
        or transition.get('source_receipt')!=proof.get('source_receipt')
        or declared.get('source_portuguese_token_kl_multiplier')!=1 or declared.get('active_portuguese_token_kl_multiplier')!=5
        or declared.get('PT_token_KL_before')!=40 or declared.get('PT_token_KL_after')!=200
        or declared.get('preservation_coefficients_by_language')!={'pt':[200,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
        or declared.get('source_actual_retention_failures')!=2
        or declared.get('source_actual_selected_step')!=8250 or declared.get('base_token_KL')!=40
        or declared.get('duration_KL')!=2 or declared.get('feature_distillation')!=20
        or declared.get('all207_Adam_sampler_RNG_history_preserved') is not True
        or declared.get('renewed_Portuguese_grace') is not False or declared.get('fresh_optimizer_or_warmup') is not False
        or declared.get('new_counter_migration') is not False or declared.get('rewound_TR_clips')!=0
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True or proof.get('source_selected_step')!=h
        or proof.get('compared_fixed_DEV_utterances')!=350 or proof.get('all_raw_transcripts_and_emitted_paths_exact') is not True
        or proof.get('all_seven_ASR_counts_exact') is not True or proof.get('all_six_retention_gates_passed') is not False):
        raise ValueError('Only measured PTtoken40to200 from explicit unqualifiedLATEST8750/counter2 admitted')
    receipt=_checked_parent_json(proof['source_receipt']);review=_checked_parent_json(declared['ROOT_review'])
    parent=_checked_parent_json(receipt['artifacts']['run_config'])
    if (receipt.get('kind')!='ROOT_owned_closed_TSC_uniform_PT_token_constraint_v1'
        or receipt.get('chosen_step')!=h or receipt.get('completed_steps')!=h or receipt.get('chosen_checkpoint_role')!='latest'
        or receipt.get('ROOT_review')!=declared['ROOT_review']
        or transition.get('parent_config_identity_sha256')!=_mirror_config_sha(parent['config_identity'])
        or parent.get('phase')!=PT_FOCUSED_PHASE or parent.get('source_step')!=8250
        or review.get('kind')!='ROOT_delegated_measured_TSC_PT_token_constraint_review_v1' or review.get('passed') is not True
        or review.get('chosen_step')!=h or review.get('completed_steps')!=h or review.get('chosen_checkpoint_role')!='latest'
        or review.get('allow_declared_PT_token_constraint_fork') is not True
        or review.get('source_chosen_is_all_six_count_qualified') is not False or review.get('source_old_five_count_qualified') is not True
        or review.get('source_counter_preserved')!=2 or review.get('source_selected_BEST_preserved')!=8250
        or review.get('source_unqualified_latest_used') is not True or review.get('source_weights_and_Adam_from_same_complete_checkpoint') is not True
        or review.get('source_moments_sampler_RNG_history_preserved') is not True or review.get('renewed_Portuguese_grace') is not False
        or review.get('explicit_coverage_rewind_clips')!=0 or review.get('no_hybrid_training_state_used') is not True):
        raise ValueError('Actual coherent latest8750/fail2 and distinct inheritedBEST8250 must remain explicit')
    change=review.get('declared_change',{});before=parent['config_identity']['training_arguments'];after=identity.get('training_arguments',{})
    # The sole new scalar may be recorded in args or only its strict transition.
    expected={**before,'portuguese_token_kl_multiplier':5.}
    if (change.get('PT_token_KL_before')!=40 or change.get('PT_token_KL_after')!=200
        or change.get('old_five_token_KL_unchanged')!=40*5/3 or change.get('feature_distillation_unchanged')!=20
        or change.get('duration_KL_unchanged')!=2 or change.get('head_LR_unchanged')!=1e-5 or change.get('LoRA_LR_unchanged')!=1e-4
        or change.get('TR_per_update')!=14 or change.get('PT_replay_per_update')!=1 or change.get('old_five_replay_per_update')!=1
        or change.get('fixed_integer_gate_limits_unchanged') is not True or after not in (before,expected)
        or config.get('portuguese_token_kl_multiplier')!=5
        or identity.get('domain_sampling_identity')!=parent['config_identity'].get('domain_sampling_identity')
        or identity.get('retention_constraint_transition')!=parent['config_identity'].get('retention_constraint_transition')
        or config.get('retention_constraint_transition')!=parent.get('retention_constraint_transition')
        or identity.get('PT_replay_transition')!=parent['config_identity'].get('PT_replay_transition')
        or config.get('PT_replay_policy_transition')!=parent.get('PT_replay_policy_transition')
        or config.get('retention_numeric_correction')!=parent.get('retention_numeric_correction')
        or identity.get('retention_numeric_correction')!=parent['config_identity'].get('retention_numeric_correction')
        or config.get('steps')!=parent.get('steps') or config.get('remaining_uniform_pass_updates')!=config['steps']-h
        or config.get('uniform_full_pass')!=parent.get('uniform_full_pass') or config.get('Portuguese_recovery')!=parent.get('Portuguese_recovery')
        or config.get('inherited_history_steps')!=h or config.get('majority_source_step')!=7621 or config.get('uniform_origin_step')!=6500
        or config.get('trainable_parameter_tensors')!=207 or config.get('replay_clips_used')!=1914
        or required_eval_languages(config)!=required_eval_languages(parent)):
        raise ValueError('PTtoken multiplier is the sole delta; Feature20/LRs/selector/all207/horizon/grace preserved')
    gates=proof['source_retention_gates']
    if (any(gates[language].get('passed') is not True for language in ('en','es','it','fr','de'))
        or gates['pt'].get('passed') is not False or gates['pt'].get('word_errors')!=70 or gates['pt'].get('allowed_word_errors')!=66):
        raise ValueError('Actual unqualified sourcePT70/66 andoldfive pass required, not inheritedBEST claims')
    contexts=feature_phase_contexts_for(parent,history);base=next((row for row in history if row.get('step')==h),None)
    if base is None:raise ValueError('Actual8750 full-state completed-update baseline required')
    feature_phase_value_for_record(base,contexts,{})
    return {'step':h,'budget':config['steps']-h,'contexts':contexts,'source_gates':gates,
        'baseline':{key:base[key] for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds')}}


def PT_token_phase_baseline_steps(config):
    return {6500,7621,7750,8000,8250,config['source_step']} if config.get('PT_token_retention_constraint') is not None else feature_phase_baseline_steps(config)


def PT_token_phase_initial_development_step(config):
    return config['source_step'] if config.get('PT_token_retention_constraint') is not None else feature_phase_initial_development_step(config)


def PT_token_phase_actual_development_row(config,record):
    if config.get('PT_token_retention_constraint') is not None:
        return 'validation' in record and record['step']>config['source_step'] and record.get('PT_focus_updates')==record['step']-8000
    return feature_phase_actual_development_row(config,record)


def PT_token_phase_contexts_for(config,baseline_rows):
    context=PT_token_context(config,baseline_rows)
    if context is None:return feature_phase_contexts_for(config,baseline_rows)
    contexts=context['contexts'].copy();contexts['PT_token']=context;return contexts


def PT_token_phase_value_for_record(record,contexts,development):
    context=contexts.get('PT_token')
    if context is None:return feature_phase_value_for_record(record,contexts,development)
    h=context['step'];n=record['step']-h;checked=record
    if n>0:
        if (record.get('preservation_coefficients_by_language')!={'pt':[200,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
            or record.get('Portuguese_token_KL_multiplier')!=5 or record.get('PT_token_constraint_updates')!=n
            or record.get('replay_kl_coefficient')!=40 or record.get('duration_kl_coefficient')!=2 or record.get('feature_distillation_coefficient')!=20):
            raise ValueError('Actualupdate must implement PTtoken200 only; oldfive/feature/duration unchanged')
        checked={**record,'preservation_coefficients_by_language':{'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}}
    values=feature_phase_value_for_record(checked,contexts,development)
    if n<=0:
        values={key:value for key,value in values.items() if not key.startswith(('dev/','retention/'))}
        values['phase/inherited_history']=int(n<0)
        if n==0:
            actual=development.get(h,{})
            if set(actual)=={'dev/wer_'+l+'_percent' for l in ('tr','en','es','it','fr','de','pt')}:
                values.update(actual);values.update({'retention/all_dev_gates_passed':0,'retention/all_six_passed':0,'retention/original_five_passed':1,
                    'retention/Portuguese_stop_grace_active':0,'retention/stop_guard_passed':0})
    if n<0:return values
    if n>context['budget']:raise ValueError('PTtoken continuation exceeds same-pass horizon')
    values.update({'phase/PT_token_constraint_updates':n,'phase/PT_token_constraint_source_step':h,
        'phase/Portuguese_token_KL_multiplier':5.,'phase/source_Portuguese_token_KL_coefficient':40.,
        'phase/PT_active_replay_KL_coefficient':200.,'phase/PT_expected_KL_mean_density':200/16,
        'phase/PT_expected_token_density_multiplier_vs_PTfocus':5.,
        'phase/expected_RP_replay_KL_mean_density':(200+40*5/3)/16,
        'phase/PT_source_LATEST_is_all_six_qualified':0,'phase/inherited_qualified_BEST_step':8250,
        'phase/source_chosen_is_coherent_BEST_not_hybrid':0,'phase/source_chosen_is_coherent_LATEST_not_hybrid':1,
        'phase/feature_8250_ancestor_was_coherent_BEST':1,'phase/feature_8250_ancestor_rewind_clips':7000,
        'phase/source_actual_retention_failure_counter':2,'phase/explicit_failed_branch_TR_clip_rewind':0,
        'retention/source_H_Portuguese_word_errors':70,'retention/source_H_all_six_qualified':0,
        'retention/PT_token_source_all_six_qualified':0,'retention/PT_token_source_actual_failure_counter':2})
    if n==0:return values
    delta={key:record[key]-value for key,value in context['baseline'].items()}
    if any(not finite_number(value) or value<=0 for value in delta.values()) or not math.isclose(delta['audio_seen_seconds'],delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('Actualpost8750 complete-update counters required')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_PT_token_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_PT_token_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_PT_token_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_PT_token_train_wall_seconds':wall})
    return values


def PT_token_selection_values(config,selected,last_step):
    if config.get('PT_token_retention_constraint') is None:return feature_selection_values(config,selected,last_step)
    if selected.get('run_id')!=config.get('run_id'):return {}
    h=config['source_step'];step=selected.get('completed_steps');n=selected.get('PT_focus_updates')
    if type(step) is not int or step<h or step>config['steps'] or selected.get('source_step')!=h or n!=step-8000:
        raise ValueError('PTtoken source8750 versus carriedPT8000 selection clocks differ')
    inherited={**config,'source_step':8250,'remaining_uniform_pass_updates':config['steps']-8250}
    values=feature_selection_values(inherited,{**selected,'source_step':8250},last_step)
    values.update({'selection/PT_token_source_step':h,'selection/published_PT_token_constraint_updates':step-h})
    if selected['accepted']:values['selection/eligible_BEST_inherited_at_or_before_H']=int(selected['selected_step']<=h)
    return values
