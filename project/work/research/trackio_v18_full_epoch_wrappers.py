def full_epoch_context(config,history):
    transition=config.get('full_epoch_policy_transition')
    if transition is None:return None
    identity=config.get('config_identity',{});proof=config.get('initial_source_equivalence',{});declared=transition.get('declared',{});h=config.get('source_step')
    if (config.get('phase')!=PT_FOCUSED_PHASE or h!=9000 or config.get('PT_focus_source_step')!=8000
        or identity.get('full_epoch_policy_transition')!=transition
        or transition.get('kind')!='TSC_uniform_USER_full_epoch_monitor_only_full_state_v1' or transition.get('source_step')!=h
        or transition.get('source_receipt')!=proof.get('source_receipt')
        or declared.get('source_portuguese_token_kl_multiplier')!=5 or declared.get('active_portuguese_token_kl_multiplier')!=1
        or declared.get('PT_token_KL_before')!=200 or declared.get('PT_token_KL_after')!=40
        or declared.get('source_retention_stop_policy')!='enforced' or declared.get('active_retention_stop_policy')!='monitor_only'
        or declared.get('retention_gates_for_champion_selection_unchanged') is not True
        or declared.get('retention_failure_counters_and_history_preserved') is not True
        or declared.get('retention_early_stops_disabled_before_actual_full_unique_pass') is not True
        or declared.get('resource_and_nonfinite_protections_unchanged') is not True
        or declared.get('preservation_coefficients_by_language')!={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
        or declared.get('source_actual_retention_failures')!=3 or declared.get('source_actual_selected_step')!=8250
        or declared.get('new_counter_migration') is not False or declared.get('rewound_TR_clips')!=0
        or declared.get('full_pass_requires_actual_unique_TR_clips')!=253798
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True or proof.get('source_selected_step')!=h
        or proof.get('compared_fixed_DEV_utterances')!=350 or proof.get('all_raw_transcripts_and_emitted_paths_exact') is not True
        or proof.get('all_seven_ASR_counts_exact') is not True or proof.get('all_six_retention_gates_passed') is not False):
        raise ValueError('Explicit USER full-epoch monitor-only/PT5to1/latest9000/counter3 required')
    receipt=_checked_parent_json(proof['source_receipt']);review=_checked_parent_json(declared['ROOT_review'])
    parent=_checked_parent_json(receipt['artifacts']['run_config'])
    if (receipt.get('kind')!='ROOT_owned_closed_TSC_uniform_full_epoch_v1'
        or receipt.get('chosen_step')!=h or receipt.get('completed_steps')!=h or receipt.get('chosen_checkpoint_role')!='latest'
        or receipt.get('ROOT_review')!=declared['ROOT_review']
        or transition.get('parent_config_identity_sha256')!=_mirror_config_sha(parent['config_identity'])
        or parent.get('phase')!=PT_FOCUSED_PHASE or parent.get('source_step')!=8750
        or review.get('kind')!='ROOT_USER_TSC_full_epoch_policy_review_v1' or review.get('passed') is not True
        or review.get('chosen_step')!=h or review.get('completed_steps')!=h or review.get('chosen_checkpoint_role')!='latest'
        or review.get('allow_full_epoch_monitor_only_continuation') is not True
        or review.get('source_counter_preserved')!=3 or review.get('source_selected_BEST_preserved')!=8250
        or review.get('source_unqualified_latest_used') is not True or review.get('no_hybrid_training_state_used') is not True
        or review.get('explicit_coverage_rewind_clips')!=0 or review.get('renewed_Portuguese_grace') is not False
        or review.get('source_weights_and_Adam_from_same_complete_checkpoint') is not True
        or review.get('source_moments_sampler_RNG_history_preserved') is not True):
        raise ValueError('Actual user-approved coherentlatest9000, unchangedcounter3/BEST8250 required')
    change=review.get('declared_change',{});expected={**parent['config_identity']['training_arguments'],
        'portuguese_token_kl_multiplier':1.,'retention_stop_policy':'monitor_only'}
    if (change.get('source_portuguese_token_kl_multiplier')!=5 or change.get('active_portuguese_token_kl_multiplier')!=1
        or change.get('PT_token_KL_before')!=200 or change.get('PT_token_KL_after')!=40
        or change.get('active_retention_stop_policy')!='monitor_only'
        or change.get('strict_all_six_integer_champion_gates_unchanged') is not True
        or change.get('memory_nonfinite_crash_protections_unchanged') is not True
        or change.get('no_plateau_target_or_retention_stop_before_complete_unique_pass') is not True
        or change.get('normal_budget_windows_automatically_exact_resume_despite_retention_failures') is not True
        or identity.get('training_arguments')!=expected or config.get('portuguese_token_kl_multiplier')!=1
        or config.get('retention_stop_policy')!='monitor_only'
        or identity.get('domain_sampling_identity')!=parent['config_identity'].get('domain_sampling_identity')
        or any(identity.get(key)!=parent['config_identity'].get(key) for key in
            ('retention_constraint_transition','PT_token_retention_constraint','PT_replay_transition','retention_numeric_correction'))
        or any(config.get(key)!=parent.get(key) for key in
            ('retention_constraint_transition','PT_token_retention_constraint','PT_replay_policy_transition','retention_numeric_correction','uniform_full_pass','Portuguese_recovery'))
        or config.get('steps')!=parent.get('steps') or config.get('remaining_uniform_pass_updates')!=config['steps']-h
        or config.get('inherited_history_steps')!=h or config.get('majority_source_step')!=7621 or config.get('uniform_origin_step')!=6500
        or config.get('trainable_parameter_tensors')!=207 or config.get('replay_clips_used')!=1914
        or required_eval_languages(config)!=required_eval_languages(parent)):
        raise ValueError('Only explicit stop-policy andPTrollback delta; scientificdata/LRs/207/queues/horizon preserved')
    gates=proof['source_retention_gates']
    if gates['pt'].get('word_errors')!=71 or gates['pt'].get('passed') is not False or gates['de'].get('word_errors')!=71 or gates['de'].get('passed') is not False:
        raise ValueError('Initial9000 sourceDE/PT71 remainfalse, not fabricatedqualifiedsource')
    contexts=PT_token_phase_contexts_for(parent,history);base=next((row for row in history if row.get('step')==h),None)
    if base is None:raise ValueError('Actual9000 full-state completed-update baseline required')
    PT_token_phase_value_for_record(base,contexts,{})
    return {'step':h,'budget':config['steps']-h,'contexts':contexts,'source_gates':gates,
        'baseline':{key:base[key] for key in ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds')}}


def full_epoch_phase_baseline_steps(config):
    return {6500,7621,7750,8000,8250,8750,config['source_step']} if config.get('full_epoch_policy_transition') is not None else PT_token_phase_baseline_steps(config)


def full_epoch_phase_initial_development_step(config):
    return config['source_step'] if config.get('full_epoch_policy_transition') is not None else PT_token_phase_initial_development_step(config)


def full_epoch_phase_actual_development_row(config,record):
    if config.get('full_epoch_policy_transition') is not None:
        return 'validation' in record and record['step']>config['source_step'] and record.get('full_epoch_continuation_updates')==record['step']-config['source_step']
    return PT_token_phase_actual_development_row(config,record)


def full_epoch_phase_contexts_for(config,baseline_rows):
    context=full_epoch_context(config,baseline_rows)
    if context is None:return PT_token_phase_contexts_for(config,baseline_rows)
    contexts=context['contexts'].copy();contexts['full_epoch']=context;return contexts


def full_epoch_phase_value_for_record(record,contexts,development):
    context=contexts.get('full_epoch')
    if context is None:return PT_token_phase_value_for_record(record,contexts,development)
    h=context['step'];n=record['step']-h;checked=record
    if n>0:
        if (n>context['budget'] or record.get('full_epoch_continuation_updates')!=n
            or record.get('retention_stop_policy')!='monitor_only' or record.get('Portuguese_token_KL_multiplier')!=1
            or record.get('preservation_coefficients_by_language')!={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}):
            raise ValueError('Actualnewupdate PTrollback/monitor-only fields differ')
        gate=record.get('validation')
        if gate is not None and gate.get('retention_early_stop_active') is not False:
            raise ValueError('Declaredmonitor-only retention remainsinactive for training')
        checked={**record,'Portuguese_token_KL_multiplier':5.,
            'preservation_coefficients_by_language':{'pt':[200,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}}
    values=PT_token_phase_value_for_record(checked,contexts,development)
    if n<=0:
        values={key:value for key,value in values.items() if not key.startswith(('dev/','retention/'))}
        values['phase/inherited_history']=int(n<0)
        if n==0:
            actual=development.get(h,{})
            if set(actual)=={'dev/wer_'+l+'_percent' for l in ('tr','en','es','it','fr','de','pt')}:
                values.update(actual);values.update({'retention/all_dev_gates_passed':0,'retention/all_six_passed':0,
                    'retention/original_five_passed':0,'retention/Portuguese_stop_grace_active':0,'retention/stop_guard_passed':0})
    if n<0:return values
    values.update({'phase/full_epoch_continuation_updates':n,'phase/full_epoch_source_step':h,
        'phase/retention_stop_policy_monitor_only':1,'phase/strict_all_six_champion_selection_unchanged':1,
        'phase/full_pass_requires_actual_unique_TR_clips':253798,'phase/Portuguese_token_KL_multiplier':1.,
        'phase/PT_active_replay_KL_coefficient':40.,'phase/PT_expected_KL_mean_density':40/16,
        'phase/PT_expected_token_density_multiplier_vs_PTfocus':1.,'phase/expected_RP_replay_KL_mean_density':(40+40*5/3)/16,
        'phase/prior_PT_token_constraint_attempted_multiplier':5.,'phase/source_actual_retention_failure_counter':3,
        'phase/inherited_qualified_BEST_step':8250,'phase/explicit_failed_branch_TR_clip_rewind':0,
        'retention/source_H_Portuguese_word_errors':71,'retention/source_H_all_six_qualified':0,
        'retention/full_epoch_source_actual_failure_counter':3,'retention/early_stop_active':0,
        'retention/training_monitor_only_policy':1,'retention/champion_selection_all_six_gates_required':1})
    if n==0:return values
    delta={key:record[key]-value for key,value in context['baseline'].items()}
    if any(not finite_number(value) or value<=0 for value in delta.values()) or not math.isclose(delta['audio_seen_seconds'],delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('Actualpost9000 complete-update counters required')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_full_epoch_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_full_epoch_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_full_epoch_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_full_epoch_train_wall_seconds':wall})
    return values


def full_epoch_selection_values(config,selected,last_step):
    if config.get('full_epoch_policy_transition') is None:return PT_token_selection_values(config,selected,last_step)
    if selected.get('run_id')!=config.get('run_id'):return {}
    h=config['source_step'];step=selected.get('completed_steps')
    if type(step) is not int or step<h or step>config['steps'] or selected.get('source_step')!=h:
        raise ValueError('Full-epoch source9000 andpublishedfrontier differ')
    inherited={**config,'source_step':8750,'remaining_uniform_pass_updates':config['steps']-8750}
    values=PT_token_selection_values(inherited,{**selected,'source_step':8750},last_step)
    values.update({'selection/full_epoch_source_step':h,'selection/published_full_epoch_continuation_updates':step-h,
        'selection/retention_failures_monitor_only':1,'selection/strict_all_six_champion_gates_unchanged':1})
    return values


def full_epoch_rejected_run_terminal(config,selected):
    if config.get('full_epoch_policy_transition') is not None:
        return selected.get('full_new_uniform_pass_complete') is True
    return ptfocus_rejected_run_terminal(config,selected)
