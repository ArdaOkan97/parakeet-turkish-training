        parent_feature=d/'parent_feature.json';write_json(parent_feature,feature_config)
        PT_review=d/'PT_review.json'
        PT_measurement={'kind':'ROOT_delegated_measured_TSC_PT_token_constraint_review_v1','passed':True,'chosen_step':8750,'completed_steps':8750,'chosen_checkpoint_role':'latest',
            'allow_declared_PT_token_constraint_fork':True,'source_chosen_is_all_six_count_qualified':False,'source_old_five_count_qualified':True,
            'source_counter_preserved':2,'source_selected_BEST_preserved':8250,'source_unqualified_latest_used':True,
            'source_weights_and_Adam_from_same_complete_checkpoint':True,'source_moments_sampler_RNG_history_preserved':True,'renewed_Portuguese_grace':False,
            'explicit_coverage_rewind_clips':0,'no_hybrid_training_state_used':True,
            'declared_change':{'PT_token_KL_before':40.,'PT_token_KL_after':200.,'old_five_token_KL_unchanged':40*5/3,
                'feature_distillation_unchanged':20.,'duration_KL_unchanged':2.,'head_LR_unchanged':1e-5,'LoRA_LR_unchanged':1e-4,
                'TR_per_update':14,'PT_replay_per_update':1,'old_five_replay_per_update':1,'fixed_integer_gate_limits_unchanged':True}}
        write_json(PT_review,PT_measurement)
        PT_receipt=d/'PT_receipt.json';write_json(PT_receipt,{'kind':'ROOT_owned_closed_TSC_uniform_PT_token_constraint_v1','chosen_step':8750,'completed_steps':8750,
            'chosen_checkpoint_role':'latest','ROOT_review':record(PT_review),'artifacts':{'run_config':record(parent_feature)}})
        PT_declared={'source_portuguese_token_kl_multiplier':1.,'active_portuguese_token_kl_multiplier':5.,'PT_token_KL_before':40.,'PT_token_KL_after':200.,
            'base_token_KL':40.,'duration_KL':2.,'feature_distillation':20.,'source_actual_retention_failures':2,'source_actual_selected_step':8250,
            'preservation_coefficients_by_language':{'pt':[200,2,20],'old_five':[40*5/3,2*5/3,20*5/3]},'ROOT_review':record(PT_review),
            'all207_Adam_sampler_RNG_history_preserved':True,'renewed_Portuguese_grace':False,'fresh_optimizer_or_warmup':False,'new_counter_migration':False,'rewound_TR_clips':0}
        PT_transition={'kind':'TSC_uniform_PT_token_retention_constraint_full_state_v1','source_step':8750,'source_receipt':record(PT_receipt),
            'parent_config_identity_sha256':ns['_mirror_config_sha'](feature_config['config_identity']),'declared':PT_declared}
        source_gates={**corrected_gates,'pt':{**corrected_gates['pt'],'word_errors':70,'passed':False}}
        PT_config={**feature_config,'source_step':8750,'inherited_history_steps':8750,'remaining_uniform_pass_updates':16359,'portuguese_token_kl_multiplier':5.,
            'PT_token_retention_constraint':PT_transition,
            'config_identity':{**feature_config['config_identity'],'PT_token_retention_constraint':PT_transition,
                'training_arguments':{**feature_config['config_identity']['training_arguments'],'portuguese_token_kl_multiplier':5.}},
            'initial_source_equivalence':{**feature_config['initial_source_equivalence'],'source_selected_step':8750,'source_receipt':record(PT_receipt),
                'all_six_retention_gates_passed':False,'source_retention_gates':source_gates}}
        latest8750={**live_row,'step':8750,'PT_focus_updates':750,'audio_seen_seconds':35000.,'turkish_audio_seen_seconds':24500.,'replay_audio_seen_seconds':10500.,'train_wall_seconds':1300.,
            'replay_language_clip_counts':{**row['replay_language_clip_counts'],'pt':850,'en':930},
            'Turkish_domain_coverage':{'PT_focus_transition':{'source_step':8000,'local_updates':750,'Portuguese_draws_after_transition':750,'old_five_draws_after_transition':750}}}
        PT_contexts=ns['PT_token_phase_contexts_for'](PT_config,[base,row,latest8750])
        newPT={**latest8750,'step':8760,'PT_focus_updates':760,'Portuguese_token_KL_multiplier':5.,'PT_token_constraint_updates':10,
            'audio_seen_seconds':35200.,'turkish_audio_seen_seconds':24640.,'replay_audio_seen_seconds':10560.,'train_wall_seconds':1310.,
            'preservation_coefficients_by_language':{'pt':[200,2,20],'old_five':[40*5/3,2*5/3,20*5/3]},
            'replay_language_clip_counts':{**latest8750['replay_language_clip_counts'],'pt':860,'en':940},
            'Turkish_domain_coverage':{'PT_focus_transition':{'source_step':8000,'local_updates':760,'Portuguese_draws_after_transition':760,'old_five_draws_after_transition':760}}}
        raw=json.dumps(newPT,sort_keys=True);PT_values=ns['PT_token_phase_value_for_record'](newPT,PT_contexts,{})
        ok(json.dumps(newPT,sort_keys=True)==raw and PT_values['phase/PT_token_constraint_updates']==10 and PT_values['phase/PT_focus_updates']==760
            and PT_values['phase/feature_constraint_updates']==510 and PT_values['resources/post_PT_token_audio_minutes_per_wall_minute']==20,
            'Source8750 rates/localN distinctfrom8250/8000 clocks; rawrowunchanged')
        ok(PT_values['phase/PT_active_replay_KL_coefficient']==200 and PT_values['phase/PT_expected_KL_mean_density']==12.5
            and PT_values['phase/each_old_five_active_replay_KL_coefficient']==40*5/3
            and PT_values['phase/PT_active_feature_coefficient']==20 and PT_values['phase/each_old_five_active_feature_coefficient']==20*5/3,
            'OnlyPTtoken5x; active oldfive/feature/duration unchanged')
        initialPT=ns['PT_token_phase_value_for_record'](latest8750,PT_contexts,{8750:{'dev/wer_'+l+'_percent':40. for l in ('tr','en','es','it','fr','de','pt')}})
        ok(initialPT['retention/all_six_passed']==0 and initialPT['retention/original_five_passed']==1
            and initialPT['phase/source_actual_retention_failure_counter']==2 and initialPT['phase/inherited_qualified_BEST_step']==8250
            and initialPT['phase/source_chosen_is_coherent_BEST_not_hybrid']==0 and initialPT['phase/explicit_failed_branch_TR_clip_rewind']==0,
            'UnqualifiedLATEST/PT70/counter2 never claimedBEST; qualified8250 distinct')
        ok(not any(k.startswith(('dev/','retention/')) for k in ns['PT_token_phase_value_for_record'](row,PT_contexts,{8750:{}})),
            'No inherited8250 DEV fabricatedasnew8750 sourcepoint')
        bad(lambda:ns['PT_token_phase_value_for_record']({**newPT,'preservation_coefficients_by_language':latest8750['preservation_coefficients_by_language']},PT_contexts,{}),'PTtoken40 cannot masqueradeactive200')
        bad(lambda:ns['PT_token_phase_value_for_record']({**newPT,'PT_token_constraint_updates':760},PT_contexts,{}),'NewlocalN mustnotresetorreplacecarried PT760')
        bad(lambda:ns['PT_token_phase_contexts_for']({**PT_config,'initial_source_equivalence':{**PT_config['initial_source_equivalence'],'all_six_retention_gates_passed':True}},[base,row,latest8750]),'FalsequalifiedLATEST flag rejected')
        corrupt={**PT_config,'PT_token_retention_constraint':{**PT_transition,'declared':{**PT_declared,'source_actual_retention_failures':0}}}
        bad(lambda:ns['PT_token_phase_contexts_for'](corrupt,[base,row,latest8750]),'Failurecounterreset rejected')
        ok(ns['PT_token_phase_baseline_steps'](PT_config)=={6500,7621,7750,8000,8250,8750}
            and ns['PT_token_phase_initial_development_step'](PT_config)==8750,'Six bounded baselines/alloldlineagepreserved')
