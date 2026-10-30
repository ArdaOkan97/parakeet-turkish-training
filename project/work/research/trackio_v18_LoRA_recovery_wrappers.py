def recovery_context(config,history):
    transition=config.get('feature_LoRA_LR_transition')
    if transition is None:return None
    identity=config.get('config_identity',{});proof=config.get('initial_source_equivalence',{});declared=transition.get('declared',{});h=config.get('source_step')
    if (config.get('phase')!=PT_FOCUSED_PHASE or h!=8250 or config.get('PT_focus_source_step')!=8000
        or identity.get('feature_LoRA_LR_transition')!=transition
        or transition.get('kind')!='TSC_uniform_feature_lower_LoRA_full_state_v1' or transition.get('source_step')!=h
        or transition.get('source_receipt')!=proof.get('source_receipt')
        or declared.get('source_lora_lr')!=1e-4 or declared.get('active_lora_lr')!=3e-5 or declared.get('source_step')!=h
        or declared.get('failed_completed_step')!=9000 or declared.get('rewound_TR_clips')!=10500
        or declared.get('chosen_uniform_unique_clips')!=17774
        or declared.get('preservation_coefficients_by_language')!={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
        or declared.get('all207_Adam_sampler_RNG_history_preserved') is not True
        or declared.get('source_actual_retention_failures')!=0 or declared.get('source_actual_selected_step')!=h
        or declared.get('source_Adam_moments_or_clock_reset') is not False or declared.get('new_counter_migration') is not False
        or declared.get('renewed_Portuguese_grace') is not False
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True or proof.get('source_selected_step')!=h
        or proof.get('compared_fixed_DEV_utterances')!=350 or proof.get('all_raw_transcripts_and_emitted_paths_exact') is not True
        or proof.get('all_seven_ASR_counts_exact') is not True or proof.get('all_six_retention_gates_passed') is not True):
        raise ValueError('Only coherent qualifiedBEST8250 encoderLR1e-4to3e-5 with explicit10500 rewind admitted')
    receipt=_checked_parent_json(proof['source_receipt']);review=_checked_parent_json(declared['ROOT_review'])
    parent=_checked_parent_json(receipt['artifacts']['run_config'])
    if (receipt.get('kind')!='ROOT_owned_closed_TSC_uniform_feature_LoRA_LR_v1'
        or receipt.get('chosen_step')!=h or receipt.get('completed_steps')!=8750 or receipt.get('chosen_checkpoint_role')!='best'
        or receipt.get('ROOT_review')!=declared['ROOT_review']
        or transition.get('parent_config_identity_sha256')!=_mirror_config_sha(parent['config_identity'])
        or parent.get('phase')!=PT_FOCUSED_PHASE or parent.get('source_step')!=h
        or review.get('kind')!='ROOT_delegated_measured_TSC_feature_LoRA_LR_review_v1' or review.get('passed') is not True
        or review.get('chosen_step')!=h or review.get('completed_steps')!=8750 or review.get('chosen_checkpoint_role')!='best'
        or review.get('allow_declared_feature_LoRA_LR_fork') is not True
        or review.get('source_counter_preserved')!=0 or review.get('source_selected_BEST_preserved')!=h
        or review.get('source_unqualified_latest_used') is not False or review.get('source_chosen_is_all_six_count_qualified') is not True
        or review.get('source_weights_and_Adam_from_same_complete_checkpoint') is not True
        or review.get('source_moments_sampler_RNG_history_preserved') is not True or review.get('source_Adam_audit_before_LR_change') is not True
        or review.get('renewed_Portuguese_grace') is not False or review.get('explicit_coverage_rewind_clips')!=10500
        or review.get('failed_branch_history_preserved') is not True or review.get('no_hybrid_training_state_used') is not True
        or declared.get('recent_failed_PT_token_review')!=review.get('recent_failed_PT_token_review')):
        raise ValueError('Actual qualifiedsource/dormantfailed9000 branch must remain explicit and hash-bound')
    failed=_checked_parent_json(declared['recent_failed_PT_token_review'])
    if failed.get('completed_steps')!=9000:raise ValueError('Actualfailed9000 review required for declared10500 rewind')
    change=review.get('declared_change',{})
    if (change.get('parameter')!='lora_lr' or change.get('source_value')!=1e-4 or change.get('active_value')!=3e-5
        or change.get('feature_distillation_unchanged')!=20 or change.get('token_KL_unchanged')!=40
        or change.get('duration_KL_unchanged')!=2 or change.get('head_LR_unchanged')!=1e-5
        or change.get('TR_per_update')!=14 or change.get('PT_replay_per_update')!=1 or change.get('old_five_replay_per_update')!=1
        or change.get('fixed_integer_gate_limits_unchanged') is not True
        or identity.get('training_arguments')!={**parent['config_identity']['training_arguments'],'lora_lr':3e-5}
        or identity.get('domain_sampling_identity')!=parent['config_identity'].get('domain_sampling_identity')
        or identity.get('retention_constraint_transition')!=parent['config_identity'].get('retention_constraint_transition')
        or config.get('retention_constraint_transition')!=parent.get('retention_constraint_transition')
        or identity.get('PT_replay_transition')!=parent['config_identity'].get('PT_replay_transition')
        or config.get('PT_replay_policy_transition')!=parent.get('PT_replay_policy_transition')
        or config.get('retention_numeric_correction')!=parent.get('retention_numeric_correction')
        or identity.get('retention_numeric_correction')!=parent['config_identity'].get('retention_numeric_correction')
        or config.get('PT_token_retention_constraint') is not None or identity.get('PT_token_retention_constraint') is not None
        or proof.get('source_retention_gates')!=parent.get('initial_source_equivalence',{}).get('source_retention_gates')
        or config.get('steps')!=parent.get('steps') or config.get('remaining_uniform_pass_updates')!=config['steps']-h
        or config.get('uniform_full_pass')!=parent.get('uniform_full_pass') or config.get('Portuguese_recovery')!=parent.get('Portuguese_recovery')
        or config.get('inherited_history_steps')!=h or config.get('majority_source_step')!=7621 or config.get('uniform_origin_step')!=6500
        or config.get('trainable_parameter_tensors')!=207 or config.get('replay_clips_used')!=1914
        or required_eval_languages(config)!=required_eval_languages(parent)):
        raise ValueError('SoleLoRaLR delta required; Feature20/PTtoken40/selector/207/counters/fullpass unchanged')
    contexts=feature_phase_contexts_for(parent,history);base=next((row for row in history if row.get('step')==h),None)
    if base is None:raise ValueError('ActualcoherentBEST8250 counter baseline required')
    feature_phase_value_for_record(base,contexts,{})
    return {'step':h,'budget':config['steps']-h,'contexts':contexts,'source_gates':proof['source_retention_gates']}


def recovery_phase_baseline_steps(config):
    return {6500,7621,7750,8000,config['source_step']} if config.get('feature_LoRA_LR_transition') is not None else PT_token_phase_baseline_steps(config)


def recovery_phase_initial_development_step(config):
    return config['source_step'] if config.get('feature_LoRA_LR_transition') is not None else PT_token_phase_initial_development_step(config)


def recovery_phase_actual_development_row(config,record):
    if config.get('feature_LoRA_LR_transition') is not None:
        return 'validation' in record and record['step']>config['source_step'] and record.get('feature_LoRA_LR_updates')==record['step']-config['source_step']
    return PT_token_phase_actual_development_row(config,record)


def recovery_phase_contexts_for(config,baseline_rows):
    context=recovery_context(config,baseline_rows)
    if context is None:return PT_token_phase_contexts_for(config,baseline_rows)
    contexts=context['contexts'].copy();contexts['LoRA_recovery']=context;return contexts


def recovery_phase_value_for_record(record,contexts,development):
    context=contexts.get('LoRA_recovery')
    if context is None:return PT_token_phase_value_for_record(record,contexts,development)
    h=context['step'];n=record['step']-h;checked=record
    if n>0:
        if (n>context['budget'] or record.get('feature_LoRA_LR_updates')!=n or record.get('feature_LoRA_LR_source_step')!=h
            or record.get('lora_lr')!=3e-5 or record.get('head_lr')!=1e-5
            or record.get('preservation_coefficients_by_language')!={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}):
            raise ValueError('Actualrecovery LR/localN/teacher coefficients differ')
        checked={**record,'lora_lr':1e-4}
    values=feature_phase_value_for_record(checked,contexts,development)
    if n<0:return values
    values.update({'phase/feature_LoRA_LR_updates':n,'phase/feature_LoRA_LR_source_step':h,
        'phase/feature_LoRA_source_group_lr':1e-4,'phase/feature_LoRA_declared_group_lr':3e-5,'phase/lora_lr':3e-5,
        'phase/source_actual_retention_failure_counter':0,'phase/source_failure_counter_is_genuine_BEST_not_reset':1,
        'phase/inherited_qualified_BEST_step':8250,'phase/explicit_failed_branch_TR_clip_rewind':10500,
        'phase/failed_branch_uniform_unique_clips':28274,'phase/chosen_source_uniform_unique_clips':17774,
        'phase/source_chosen_is_coherent_BEST_not_hybrid':1})
    if n>0:
        # Same8250 actual rate baseline as feature/integermath; newnames isolate
        # this branch without recalculating or discarding older phase rates.
        for suffix in ('audio_minutes_per_wall_minute','turkish_audio_minutes_per_wall_minute','replay_audio_minutes_per_wall_minute','train_wall_seconds'):
            values['resources/post_LoRA_recovery_'+suffix]=values['resources/post_integer_gate_'+suffix]
    return values


def recovery_selection_values(config,selected,last_step):
    if config.get('feature_LoRA_LR_transition') is None:return PT_token_selection_values(config,selected,last_step)
    values=feature_selection_values(config,selected,last_step)
    if values:values.update({'selection/feature_LoRA_LR_source_step':config['source_step'],
        'selection/published_feature_LoRA_LR_updates':selected['completed_steps']-config['source_step']})
    return values
