        # A coherent BEST8250 branch changes only the feature coefficient.
        parent_integer=d/'parent_integer.json';write_json(parent_integer,integer_config)
        measured_path=d/'measured_review.json'
        measured={'kind':'ROOT_delegated_measured_TSC_retention_constraint_review_v1','passed':True,'chosen_step':8250,'completed_steps':8750,'chosen_checkpoint_role':'best',
            'allow_declared_feature_distillation_constraint_fork':True,'source_chosen_is_all_six_count_qualified':True,'source_counter_preserved':0,
            'source_weights_and_Adam_from_same_complete_checkpoint':True,'source_moments_sampler_RNG_history_preserved':True,'renewed_Portuguese_grace':False,
            'explicit_coverage_rewind_clips':7000,'failed_branch_history_preserved':True,'no_hybrid_training_state_used':True,
            'chosen_uniform_unique_clips':17774,'failed_latest_uniform_unique_clips':24774,
            'declared_change':{'parameter':'feature_distillation','source_value':4.,'active_value':20.,'token_KL_unchanged':40.,'duration_KL_unchanged':2.,
                'head_LR_unchanged':1e-5,'LoRA_LR_unchanged':1e-4,'TR_per_update':14,'PT_replay_per_update':1,'old_five_replay_per_update':1,'fixed_integer_gate_limits_unchanged':True}}
        write_json(measured_path,measured)
        feature_receipt=d/'feature_receipt.json';write_json(feature_receipt,{'kind':'ROOT_owned_closed_TSC_uniform_retention_constraint_v1','artifacts':{'run_config':record(parent_integer)}})
        constraint={'kind':'declared_feature_preservation_strengthening_v1','source_step':8250,'parent_closed_step':8750,'chosen_checkpoint_role':'best',
            'feature_distillation_before':4.,'feature_distillation_after':20.,'rewound_uniform_unique_clips':7000,'ROOT_review':record(measured_path)}
        feature_config={**integer_config,'retention_constraint_transition':constraint,
            'config_identity':{**integer_config['config_identity'],'retention_constraint_transition':constraint,'training_arguments':{**recipe,'feature_distillation':20.}},
            'initial_source_equivalence':{**integer_config['initial_source_equivalence'],'source_receipt':record(feature_receipt)}}
        feature_contexts=ns['feature_phase_contexts_for'](feature_config,[base,row])
        live_row={**next_row,'replay_kl_coefficient':40,'duration_kl_coefficient':2,'feature_distillation_coefficient':20,'head_lr':1e-5,'lora_lr':1e-4,
            'preservation_coefficients_by_language':{'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}}
        before=json.dumps(live_row,sort_keys=True);live_values=ns['feature_phase_value_for_record'](live_row,feature_contexts,{})
        ok(json.dumps(live_row,sort_keys=True)==before and live_values['phase/feature_constraint_updates']==10
            and live_values['phase/PT_focus_updates']==260,'New feature view preservesrawrow andallcarried sampler/rate clocks')
        ok(live_values['phase/declared_feature_distillation_coefficient']==20
            and live_values['phase/PT_expected_feature_mean_density']==1.25
            and math.isclose(live_values['phase/each_old_five_expected_feature_mean_density'],5/12)
            and math.isclose(live_values['phase/expected_RP_feature_mean_density'],10/3)
            and live_values['phase/PT_expected_feature_density_multiplier']==15
            and live_values['phase/each_old_five_expected_feature_density_multiplier']==5,
            'Active feature5x perlanguage andtotaldensity truthful; KL/durationunchanged')
        ok('phase/PT_expected_preservation_density_multiplier' not in live_values
            and live_values['phase/explicit_failed_branch_TR_clip_rewind']==7000
            and live_values['resources/post_integer_gate_audio_minutes_per_wall_minute']==20,'No ambiguousall-term density multiplier; actualBESTrewind/rateanchor retained')
        bad(lambda:ns['feature_phase_value_for_record']({**live_row,'feature_distillation_coefficient':4},feature_contexts,{}),'Wrongactive featurecoefficient rejected')
        bad(lambda:ns['feature_phase_value_for_record']({**live_row,'preservation_coefficients_by_language':next_row['preservation_coefficients_by_language']},feature_contexts,{}),'Wrong perlanguagefeature scaling rejected')
        bad(lambda:ns['feature_phase_contexts_for']({**feature_config,'config_identity':{**feature_config['config_identity'],'training_arguments':{**feature_config['config_identity']['training_arguments'],'lora_lr':3e-4}}},[base,row]),'Secondscientific LRchange rejected')
        bad(lambda:ns['feature_phase_contexts_for']({**feature_config,'retention_constraint_transition':{**constraint,'chosen_checkpoint_role':'latest'}},[base,row]),'UnqualifiedLATESTnot masqueradingcoherentBEST')
        ok(ns['feature_phase_baseline_steps'](feature_config)=={6500,7621,7750,8000,8250}
            and ns['feature_phase_initial_development_step'](feature_config)==8250,'Samebounded baselines/actualsource350 only')
