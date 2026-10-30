        # New namespace repeats no selector migration. Historical8250 false
        # gates stay in the old run; new source350 proves corrected counts.
        parent_PT=d/'parent_PT.json';write_json(parent_PT,config)
        counter_migration={'previous_actual_failure_counter':5,'new_counter_after_correctly_qualified_last_DEV':0,
            'correctly_eligible_source_step':8250,'weights_all207_Adam_sampler_RNG_history_LR_loss_data_mixture_coverage_preserved':True,
            'historical_false_gate_and_selection_are_preserved':True}
        specifications={'en':(63,1045,60),'es':(46,1289,50),'it':(25,1158,24),'fr':(68,1072,63),'de':(64,1004,70),'pt':(60,1095,66)}
        strict={l:{'base_errors':e,'reference_words':n,'candidate_errors':c,'allowed_errors':e+max(n//200,e//10),'passed':True} for l,(e,n,c) in specifications.items()}
        corrected_gates={l:{'word_errors':v['candidate_errors'],'reference_words':v['reference_words'],'allowed_word_errors':v['allowed_errors'],'passed':True} for l,v in strict.items()}
        review_path=d/'ROOT_review.json';write_json(review_path,{'passed':True,'source_step':8250,'allow_count_equivalent_numerical_gate_correction':True,
            'thresholds_relaxed':False,'strict_count_gates':strict,'declared_state_accounting_migration':counter_migration})
        integer_receipt=d/'integer_receipt.json';write_json(integer_receipt,{'kind':'ROOT_owned_closed_TSC_PT_focus_integer_gate_continuation_v1','source_step':8250,
            'allow_count_equivalent_numerical_gate_correction':True,'thresholds_relaxed':False,'ROOT_review':record(review_path),
            'artifacts':{'run_config':record(parent_PT)},'declared_state_accounting_migration':counter_migration})
        correction={'source_step':8250,'rule':'E+max(N//200,E//10)','ROOT_review':record(review_path),'historical_false_gates_preserved':True,
            'source_failure_counter':5,'corrected_failure_counter':0,'qualified_initial_BEST':8250}
        new_declared={**declared,'source_step':8250,'PT_focus_origin_step':8000,'thresholds_relaxed':False,
            'historical_false_gate_records_rewritten':False,'numeric_gate_migration':counter_migration}
        integer_config={**config,'source_step':8250,'PT_focus_source_step':8000,'inherited_history_steps':8250,'remaining_uniform_pass_updates':16859,
            'retention_numeric_correction':correction,'PT_replay_policy_transition':new_declared,
            'config_identity':{**config['config_identity'],'retention_numeric_correction':correction,'PT_replay_transition':{
                **config['config_identity']['PT_replay_transition'],'source_receipt':record(integer_receipt),'declared':new_declared,
                'parent_config_identity_sha256':ns['_mirror_config_sha'](config['config_identity'])}},
            'initial_source_equivalence':{**config['initial_source_equivalence'],'source_selected_step':8250,'source_receipt':record(integer_receipt),
                'source_retention_gates':corrected_gates,'all_six_retention_gates_passed':True}}
        actual_contexts=ns['integer_phase_contexts_for'](integer_config,[base,row])
        next_row={**row,'step':8260,'PT_focus_updates':260,'audio_seen_seconds':25200.,'turkish_audio_seen_seconds':17640.,'replay_audio_seen_seconds':7560.,'train_wall_seconds':1110.,
            'replay_language_clip_counts':{**row['replay_language_clip_counts'],'pt':360,'en':440},
            'Turkish_domain_coverage':{'PT_focus_transition':{'source_step':8000,'local_updates':260,'Portuguese_draws_after_transition':260,'old_five_draws_after_transition':260}}}
        actual_values=ns['integer_phase_value_for_record'](next_row,actual_contexts,{})
        ok(actual_values['phase/integer_gate_continuation_updates']==10 and actual_values['phase/PT_focus_updates']==260
            and actual_values['phase/lora_lr_fork_updates']==510 and actual_values['phase/majority_updates']==639,
            'Source8250 localN separate from carried8000/7750/7621 clocks')
        ok(actual_values['resources/post_integer_gate_audio_minutes_per_wall_minute']==20
            and actual_values['resources/post_integer_gate_turkish_audio_minutes_per_wall_minute']==14
            and actual_values['resources/post_PT_focus_audio_minutes_per_wall_minute']!=20,'New8250 rates and original8000 rates separately preserved')
        new_dev={8250:{'dev/wer_'+l+'_percent':40. for l in ('tr','en','es','it','fr','de','pt')}}
        new_initial=ns['integer_phase_value_for_record'](row,actual_contexts,new_dev)
        ok(new_initial['retention/all_six_passed']==1 and new_initial['retention/numerically_corrected_source_failure_counter']==0
            and new_initial['phase/PT_focus_updates']==250 and new_initial['phase/integer_gate_continuation_updates']==0,
            'Only actual newsource350 qualifies initial8250; old8250 data remains immutable')
        inherited=ns['integer_phase_value_for_record'](base,actual_contexts,new_dev)
        ok(not any(k.startswith(('dev/','retention/')) for k in inherited) and inherited['phase/inherited_history']==1,'No inherited8000 DEV/retention fabricated as new continuation')
        ok(ns['integer_phase_baseline_steps'](integer_config)=={6500,7621,7750,8000,8250}
            and ns['integer_phase_initial_development_step'](integer_config)==8250
            and not ns['integer_phase_actual_development_row'](integer_config,{**row,'validation':{}})
            and ns['integer_phase_actual_development_row'](integer_config,{**next_row,'validation':{}}),'Five bounded baselines; only post8250 actual DEV selected')
        bad(lambda:ns['integer_phase_contexts_for']({**integer_config,'PT_focus_source_step':8250},[base,row]),'Wrong carried PTorigin rejected')
        bad(lambda:ns['integer_phase_contexts_for']({**integer_config,'retention_numeric_correction':{**correction,'corrected_failure_counter':1}},[base,row]),'Undeclared numeric correction rejected')
        corrupt={**integer_config,'initial_source_equivalence':{**integer_config['initial_source_equivalence'],'source_retention_gates':{**corrected_gates,'pt':{**corrected_gates['pt'],'word_errors':67}}}}
        bad(lambda:ns['integer_phase_contexts_for'](corrupt,[base,row]),'Count67 cannot qualify limit66')
        bad(lambda:ns['integer_phase_value_for_record']({**next_row,'PT_focus_updates':10},actual_contexts,{}),'Resetting carried PT250 history tolocal10 rejected')
        delegated=ns['ptfocus_selection_values'];ns['ptfocus_selection_values']=lambda c,s,last:{'selection/PT_focus_source_step':c['source_step']}
        selected={'run_id':'new','source_step':8250,'completed_steps':8260,'PT_focus_updates':260,'accepted':True,'selected_step':8250}
        selected_values=ns['integer_selection_values']({**integer_config,'run_id':'new'},selected,8260)
        ok(selected_values['selection/published_integer_gate_continuation_updates']==10 and selected_values['selection/PT_focus_source_step']==8000
            and selected_values['selection/eligible_BEST_inherited_at_or_before_H']==1,'New ownselection uses source8250 andcarried PT8000 separately')
        ns['ptfocus_selection_values']=delegated
