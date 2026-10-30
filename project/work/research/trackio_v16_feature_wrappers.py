def feature_constraint_context(config,history):
    transition=config.get('retention_constraint_transition')
    if transition is None:return None
    identity=config.get('config_identity',{});proof=config.get('initial_source_equivalence',{})
    declared=transition.get('declared',{});h=config.get('source_step')
    if (config.get('phase')!=PT_FOCUSED_PHASE or h!=8250 or config.get('PT_focus_source_step')!=8000
        or identity.get('retention_constraint_transition')!=transition
        or transition.get('kind')!='TSC_uniform_feature_retention_constraint_full_state_v1' or transition.get('source_step')!=h
        or transition.get('source_receipt')!=proof.get('source_receipt')
        or transition.get('fresh_optimizer_or_warmup') is not False or transition.get('Portuguese_grace_reset') is not False
        or declared.get('source_step')!=h or declared.get('failed_completed_step')!=8750
        or declared.get('source_feature_distillation')!=4
        or declared.get('active_feature_distillation')!=20 or declared.get('rewound_TR_clips')!=7000
        or declared.get('preservation_coefficients_by_language')!={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
        or declared.get('all207_Adam_sampler_RNG_history_preserved') is not True
        or declared.get('original_integer_gate_correction_preserved') is not True or declared.get('new_counter_migration') is not False
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True
        or proof.get('source_selected_step')!=h or proof.get('compared_fixed_DEV_utterances')!=350
        or proof.get('all_raw_transcripts_and_emitted_paths_exact') is not True
        or proof.get('all_seven_ASR_counts_exact') is not True or proof.get('all_six_retention_gates_passed') is not True):
        raise ValueError('Only measured feature4to20 from coherent qualifiedBEST8250 is admitted')
    receipt=_checked_parent_json(proof['source_receipt']);review=_checked_parent_json(declared['ROOT_review'])
    parent=_checked_parent_json(receipt['artifacts']['run_config'])
    if (receipt.get('kind')!='ROOT_owned_closed_TSC_uniform_retention_constraint_v1'
        or receipt.get('chosen_step')!=h or receipt.get('completed_steps')!=8750 or receipt.get('chosen_checkpoint_role')!='best'
        or receipt.get('ROOT_review')!=declared['ROOT_review']
        or transition.get('parent_config_identity_sha256')!=_mirror_config_sha(parent['config_identity'])
        or declared.get('actual_diagnostic')!=review.get('actual_diagnostic')
        or parent.get('phase')!=PT_FOCUSED_PHASE or parent.get('source_step')!=h
        or review.get('kind')!='ROOT_delegated_measured_TSC_retention_constraint_review_v1' or review.get('passed') is not True
        or review.get('chosen_step')!=h or review.get('completed_steps')!=8750 or review.get('chosen_checkpoint_role')!='best'
        or review.get('allow_declared_feature_distillation_constraint_fork') is not True
        or review.get('source_chosen_is_all_six_count_qualified') is not True or review.get('source_counter_preserved')!=0
        or review.get('source_weights_and_Adam_from_same_complete_checkpoint') is not True
        or review.get('source_moments_sampler_RNG_history_preserved') is not True or review.get('renewed_Portuguese_grace') is not False
        or review.get('explicit_coverage_rewind_clips')!=7000 or review.get('failed_branch_history_preserved') is not True
        or review.get('no_hybrid_training_state_used') is not True):
        raise ValueError('Hash-bound measured decision and complete coherent source8250 required')
    change=review.get('declared_change',{})
    if (change.get('parameter')!='feature_distillation' or change.get('source_value')!=4 or change.get('active_value')!=20
        or change.get('token_KL_unchanged')!=40 or change.get('duration_KL_unchanged')!=2
        or change.get('head_LR_unchanged')!=1e-5 or change.get('LoRA_LR_unchanged')!=1e-4
        or change.get('TR_per_update')!=14 or change.get('PT_replay_per_update')!=1 or change.get('old_five_replay_per_update')!=1
        or change.get('fixed_integer_gate_limits_unchanged') is not True
        or identity.get('training_arguments')!={**parent['config_identity']['training_arguments'],'feature_distillation':20.}
        or identity.get('domain_sampling_identity')!=parent['config_identity'].get('domain_sampling_identity')
        or identity.get('PT_replay_transition')!=parent['config_identity'].get('PT_replay_transition')
        or config.get('PT_replay_policy_transition')!=parent.get('PT_replay_policy_transition')
        or config.get('retention_numeric_correction')!=parent.get('retention_numeric_correction')
        or identity.get('retention_numeric_correction')!=parent['config_identity'].get('retention_numeric_correction')
        or proof.get('source_retention_gates')!=parent.get('initial_source_equivalence',{}).get('source_retention_gates')
        or config.get('steps')!=parent.get('steps') or config.get('remaining_uniform_pass_updates')!=config['steps']-h
        or config.get('uniform_full_pass')!=parent.get('uniform_full_pass') or config.get('Portuguese_recovery')!=parent.get('Portuguese_recovery')
        or config.get('inherited_history_steps')!=h or config.get('majority_source_step')!=7621 or config.get('uniform_origin_step')!=6500
        or config.get('trainable_parameter_tensors')!=207 or config.get('replay_clips_used')!=1914
        or required_eval_languages(config)!=required_eval_languages(parent)):
        raise ValueError('Feature constraint must change only4to20; recipe/selector/clock/grace/horizon preserved')
    # Reuses the validated integer source lineage and bounded actual baselines.
    # Only coefficient validation views are projected later; disk/history stays raw.
    contexts=integer_phase_contexts_for(parent,history)
    base=next((row for row in history if row.get('step')==h),None)
    if base is None:raise ValueError('Actual coherentBEST8250 completed-update baseline required')
    integer_phase_value_for_record(base,contexts,{})
    if review.get('chosen_uniform_unique_clips')!=17774 or review.get('failed_latest_uniform_unique_clips')!=24774:
        raise ValueError('Measured7000-clip rewind must remain explicit')
    return {'step':h,'contexts':contexts,'review':declared['ROOT_review'],'source_receipt':proof['source_receipt'],
        'source_gates':proof['source_retention_gates'],'rewind_clips':7000}


def feature_phase_baseline_steps(config):
    return {6500,7621,7750,8000,config['source_step']} if config.get('retention_constraint_transition') is not None else integer_phase_baseline_steps(config)


def feature_phase_initial_development_step(config):
    return config['source_step'] if config.get('retention_constraint_transition') is not None else integer_phase_initial_development_step(config)


def feature_phase_actual_development_row(config,record):
    if config.get('retention_constraint_transition') is not None:
        return 'validation' in record and record['step']>config['source_step'] and record.get('PT_focus_updates')==record['step']-8000
    return integer_phase_actual_development_row(config,record)


def feature_phase_contexts_for(config,baseline_rows):
    context=feature_constraint_context(config,baseline_rows)
    if context is None:return integer_phase_contexts_for(config,baseline_rows)
    contexts=context['contexts'].copy();contexts['feature_constraint']=context;return contexts


def feature_phase_value_for_record(record,contexts,development):
    context=contexts.get('feature_constraint')
    if context is None:return integer_phase_value_for_record(record,contexts,development)
    h=context['step'];n=record['step']-h;checked=record
    if n>0:
        actual={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
        if (record.get('feature_distillation_coefficient')!=20 or record.get('preservation_coefficients_by_language')!=actual
            or record.get('replay_kl_coefficient')!=40 or record.get('duration_kl_coefficient')!=2
            or record.get('head_lr')!=1e-5 or record.get('lora_lr')!=1e-4):
            raise ValueError('Actual new-update coefficients/LRs must implement sole feature5x')
        checked={**record,'feature_distillation_coefficient':4,
            'preservation_coefficients_by_language':{'pt':[40,2,4],'old_five':[40*5/3,2*5/3,4*5/3]}}
    values=integer_phase_value_for_record(checked,contexts,development)
    if n<0:return values
    # Preserve old namespaces/maps and identify legacy quota provenance explicitly.
    majority=contexts.get('majority',{})
    if majority:
        values['retention/quota_7621_source_Portuguese_word_errors']=majority['source_gates']['pt']['word_errors']
        values['retention/quota_7621_source_all_six_qualified']=int(majority['source_all_six_passed'])
    values.update({'retention/feature_constraint_source_all_six_qualified':1,
        'retention/feature_constraint_source_Portuguese_word_errors':context['source_gates']['pt']['word_errors'],
        'retention/source_H_all_six_qualified':1,'retention/source_H_Portuguese_word_errors':context['source_gates']['pt']['word_errors'],
        'phase/feature_constraint_updates':n,'phase/feature_constraint_source_step':h,
        'phase/feature_distillation_source_coefficient':4.,'phase/declared_feature_distillation_coefficient':20.,
        'phase/feature_distillation_constraint_multiplier':5.,'phase/explicit_failed_branch_TR_clip_rewind':7000,
        'phase/source_chosen_is_coherent_BEST_not_hybrid':1,'phase/PT_expected_feature_mean_density':20/16,
        'phase/each_old_five_expected_feature_mean_density':(20*5/3)/5/16,
        'phase/PT_expected_feature_density_multiplier':15.,'phase/each_old_five_expected_feature_density_multiplier':5.,
        'phase/PT_KL_duration_preservation_density_multiplier':3.,'phase/each_old_five_KL_duration_preservation_density_multiplier':1.,
        'phase/expected_RP_replay_KL_mean_density':(40+40*5/3)/16,
        'phase/expected_RP_duration_KL_mean_density':(2+2*5/3)/16,
        'phase/expected_RP_feature_mean_density':(20+20*5/3)/16,
        'phase/PT_active_replay_KL_coefficient':40.,'phase/PT_active_duration_KL_coefficient':2.,'phase/PT_active_feature_coefficient':20.,
        'phase/each_old_five_active_replay_KL_coefficient':40*5/3,
        'phase/each_old_five_active_duration_KL_coefficient':2*5/3,'phase/each_old_five_active_feature_coefficient':20*5/3})
    # These old all-term multipliers cease to describe the active new phase.
    values.pop('phase/PT_expected_preservation_density_multiplier',None)
    values.pop('phase/each_old_five_expected_preservation_density_multiplier',None)
    return values


def feature_selection_values(config,selected,last_step):
    if config.get('retention_constraint_transition') is None:return integer_selection_values(config,selected,last_step)
    values=integer_selection_values(config,selected,last_step)
    if values:values.update({'selection/feature_constraint_source_step':config['source_step'],
        'selection/published_feature_constraint_updates':selected['completed_steps']-config['source_step']})
    return values
