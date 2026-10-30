def integer_gate_context(config,history):
    correction=config.get('retention_numeric_correction')
    if config.get('phase')!=PT_FOCUSED_PHASE or correction is None:return None
    identity=config.get('config_identity',{});transition=identity.get('PT_replay_transition',{})
    declared=transition.get('declared',{});h=config.get('source_step');proof=config.get('initial_source_equivalence',{})
    if (h!=8250 or config.get('PT_focus_source_step')!=8000 or config.get('inherited_history_steps')!=h
        or identity.get('retention_numeric_correction')!=correction or correction.get('source_step')!=h
        or correction.get('rule')!='E+max(N//200,E//10)' or correction.get('historical_false_gates_preserved') is not True
        or correction.get('source_failure_counter')!=5 or correction.get('corrected_failure_counter')!=0
        or correction.get('qualified_initial_BEST')!=h or declared!=config.get('PT_replay_policy_transition')
        or declared.get('source_step')!=h or declared.get('PT_focus_origin_step')!=8000
        or declared.get('thresholds_relaxed') is not False or declared.get('historical_false_gate_records_rewritten') is not False
        or proof.get('passed') is not True or proof.get('actual_GPU_decode') is not True
        or proof.get('source_selected_step')!=h or proof.get('compared_fixed_DEV_utterances')!=350
        or proof.get('all_raw_transcripts_and_emitted_paths_exact') is not True or proof.get('all_seven_ASR_counts_exact') is not True
        or proof.get('all_six_retention_gates_passed') is not True or proof.get('source_receipt')!=transition.get('source_receipt')):
        raise ValueError('Explicit8250 integer accounting correction and actual qualified source350 required')
    receipt=_checked_parent_json(transition['source_receipt']);review=_checked_parent_json(correction['ROOT_review'])
    parent=_checked_parent_json(receipt['artifacts']['run_config'])
    if (receipt.get('kind')!='ROOT_owned_closed_TSC_PT_focus_integer_gate_continuation_v1'
        or receipt.get('source_step')!=h or receipt.get('allow_count_equivalent_numerical_gate_correction') is not True
        or receipt.get('thresholds_relaxed') is not False or receipt.get('ROOT_review')!=correction['ROOT_review']
        or review.get('passed') is not True or review.get('source_step')!=h
        or review.get('allow_count_equivalent_numerical_gate_correction') is not True or review.get('thresholds_relaxed') is not False
        or parent.get('phase')!=PT_FOCUSED_PHASE or parent.get('source_step')!=8000
        or _mirror_config_sha(parent['config_identity'])!=transition.get('parent_config_identity_sha256')
        or config.get('majority_source_step')!=7621 or config.get('uniform_origin_step')!=6500
        or config.get('steps')!=parent.get('steps') or declared.get('configured_end_step')!=config.get('steps')
        or config.get('remaining_uniform_pass_updates')!=config['steps']-h
        or identity.get('training_arguments')!=parent['config_identity'].get('training_arguments')
        or identity.get('domain_sampling_identity')!=parent['config_identity'].get('domain_sampling_identity')
        or config.get('uniform_full_pass')!=parent.get('uniform_full_pass') or config.get('Portuguese_recovery')!=parent.get('Portuguese_recovery')
        or declared.get('declared_change')!=parent['PT_replay_policy_transition'].get('declared_change')
        or config.get('trainable_parameter_tensors')!=207 or config.get('replay_clips_used')!=1914
        or required_eval_languages(config)!=required_eval_languages(parent)):
        raise ValueError('Integer gate continuation must preserve actual8000 selector/recipe/quota and hash-bound review')
    migration=declared.get('numeric_gate_migration',{})
    if (migration!=receipt.get('declared_state_accounting_migration') or migration!=review.get('declared_state_accounting_migration')
        or migration.get('previous_actual_failure_counter')!=5 or migration.get('new_counter_after_correctly_qualified_last_DEV')!=0
        or migration.get('correctly_eligible_source_step')!=h or migration.get('weights_all207_Adam_sampler_RNG_history_LR_loss_data_mixture_coverage_preserved') is not True
        or migration.get('historical_false_gate_and_selection_are_preserved') is not True):
        raise ValueError('Declared counter correction must match actual ROOT review without rewriting historical flags')
    for language in ('en','es','it','fr','de','pt'):
        gate=proof['source_retention_gates'][language];checked=review['strict_count_gates'][language]
        e=checked.get('base_errors');n=checked.get('reference_words');candidate=checked.get('candidate_errors')
        if (type(e) is not int or type(n) is not int or type(candidate) is not int or e<0 or n<1 or candidate<0
            or checked.get('allowed_errors')!=e+max(n//200,e//10) or checked.get('passed') is not True
            or gate.get('reference_words')!=n or gate.get('word_errors')!=candidate
            or gate.get('allowed_word_errors')!=checked['allowed_errors'] or gate.get('passed') is not True
            or candidate>checked['allowed_errors']):raise ValueError('Actual source350 integer gate/review counts differ')
    contexts=ptfocus_phase_contexts_for(parent,history)
    base=next((row for row in history if row.get('step')==h),None)
    if base is None:raise ValueError('Actual8250 source rate baseline required')
    ptfocus_phase_value_for_record(base,contexts,{})
    return {'step':h,'budget':config['steps']-h,'contexts':contexts,'baseline':{key:base[key] for key in
        ('audio_seen_seconds','train_wall_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds')},
        'correction':correction,'source_gates':proof['source_retention_gates']}


def integer_phase_baseline_steps(config):
    return {6500,7621,7750,8000,config['source_step']} if config.get('retention_numeric_correction') is not None else ptfocus_phase_baseline_steps(config)


def integer_phase_initial_development_step(config):
    return config['source_step'] if config.get('retention_numeric_correction') is not None else ptfocus_phase_initial_development_step(config)


def integer_phase_actual_development_row(config,record):
    if config.get('retention_numeric_correction') is not None:
        return 'validation' in record and record['step']>config['source_step'] and record.get('PT_focus_updates')==record['step']-8000
    return ptfocus_phase_actual_development_row(config,record)


def integer_phase_contexts_for(config,baseline_rows):
    context=integer_gate_context(config,baseline_rows)
    if context is None:return ptfocus_phase_contexts_for(config,baseline_rows)
    contexts=context['contexts'].copy();contexts['integer_gate']=context;return contexts


def integer_phase_value_for_record(record,contexts,development):
    context=contexts.get('integer_gate')
    if context is None:return ptfocus_phase_value_for_record(record,contexts,development)
    values=ptfocus_phase_value_for_record(record,contexts,development);h=context['step'];n=record['step']-h
    if n<=0:
        values={key:value for key,value in values.items() if not key.startswith(('dev/','retention/'))}
        values['phase/inherited_history']=int(n<0)
        if n==0:
            actual=development.get(h,{})
            if set(actual)=={'dev/wer_'+language+'_percent' for language in ('tr','en','es','it','fr','de','pt')}:
                values.update(actual);values.update({'retention/all_dev_gates_passed':1,'retention/all_six_passed':1,
                    'retention/original_five_passed':1,'retention/Portuguese_stop_grace_active':0,'retention/source_H_all_six_qualified':1,
                    'retention/source_H_Portuguese_word_errors':context['source_gates']['pt']['word_errors'],
                    'retention/Portuguese_allowed_word_errors':context['source_gates']['pt']['allowed_word_errors'],
                    'retention/numerically_corrected_source_failure_counter':0})
    if n<0:return values
    values.update({'phase/integer_gate_continuation_updates':n,'phase/integer_gate_source_step':h,
        'phase/PT_focus_source_step':8000,'phase/integer_gate_correction_applied':1,
        'phase/historical_false_gate_records_preserved':1})
    if n==0:return values
    if n>context['budget']:raise ValueError('Continuation frontier exceeds unchanged full-pass horizon')
    delta={key:record[key]-value for key,value in context['baseline'].items()}
    if any(not finite_number(value) or value<=0 for value in delta.values()) or not math.isclose(delta['audio_seen_seconds'],delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError('Post8250 rate counters differ')
    wall=delta['train_wall_seconds']
    values.update({'resources/post_integer_gate_audio_minutes_per_wall_minute':delta['audio_seen_seconds']/wall,
        'resources/post_integer_gate_turkish_audio_minutes_per_wall_minute':delta['turkish_audio_seen_seconds']/wall,
        'resources/post_integer_gate_replay_audio_minutes_per_wall_minute':delta['replay_audio_seen_seconds']/wall,
        'resources/post_integer_gate_train_wall_seconds':wall})
    return values


def integer_selection_values(config,selected,last_step):
    if config.get('retention_numeric_correction') is None:return ptfocus_selection_values(config,selected,last_step)
    if selected.get('run_id')!=config.get('run_id'):return {}
    h=config['source_step'];step=selected.get('completed_steps');n=selected.get('PT_focus_updates')
    if (type(step) is not int or type(n) is not int or step<h or step>config['steps'] or n!=step-8000
        or selected.get('source_step')!=h):raise ValueError('Continuation source8250 versus PT8000 selection counters differ')
    view={**config,'source_step':8000,'remaining_uniform_pass_updates':config['steps']-8000}
    values=ptfocus_selection_values(view,{**selected,'source_step':8000},last_step)
    values.update({'selection/integer_gate_source_step':h,'selection/published_integer_gate_continuation_updates':step-h})
    if selected['accepted']:values['selection/eligible_BEST_inherited_at_or_before_H']=int(selected['selected_step']<=h)
    return values
