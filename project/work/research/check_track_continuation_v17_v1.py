"""One focused PT producer binding + tiny shard import; no live state or DB."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,ast,hashlib,json,math,os,re,tempfile,time
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
source=HERE/'track_continuation_v17.py';parent=HERE/'track_continuation_v16.py'
tree=ast.parse(source.read_text(encoding='utf-8'));old=ast.parse(parent.read_text(encoding='utf-8'))
def read_json(p,default=None):return json.loads(Path(p).read_text(encoding='utf-8')) if Path(p).exists() else default
def write_json(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False),encoding='utf-8')
ns={'Path':Path,'ROOT':ROOT,'json':json,'math':math,'hashlib':hashlib,'os':os,'re':re,'time':time,'datetime':datetime,'timezone':timezone,
    '__file__':str(source),'read_json':read_json,'write_json':write_json,'original':SimpleNamespace(metrics_for=lambda r:{})}
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
        try:ns[n.targets[0].id]=ast.literal_eval(n.value)
        except (ValueError,TypeError):pass
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) or isinstance(n,ast.ClassDef) and n.name in ('RunShardStore','StreamRunShardStore')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
checks=[];rejects=[]
def ok(v,label):
    if not v:raise AssertionError(label)
    checks.append(label)
def bad(call,label):
    try:call()
    except (ValueError,KeyError):checks.append(label);rejects.append(label);return
    raise AssertionError('Unexpected acceptance '+label)
def record(p):return {k:ns['file_record'](p)[k] for k in ('path','sha256','bytes')}
def main(args):
    assert not args.output.exists(),'Fresh proof output required'
    exempt={'follow','parser','main','publish_cycle_memory','start_controls'}
    names=[n.name for n in old.body if isinstance(n,ast.FunctionDef) and n.name not in exempt]
    ok(all(ast.dump(n,include_attributes=False)==ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==n.name),include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef) and n.name in names),'Every inherited V16 metric/math function AST unchanged')
    with tempfile.TemporaryDirectory(prefix='v17_PT_tiny_',dir=ROOT/'work/research') as d:
        d=Path(d);recipe={'head_lr':1e-5,'lora_lr':1e-4,'replay_fraction':.125,'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.}
        prior_counts={'pt':100,'en':180,'es':180,'it':180,'fr':180,'de':180}
        focus={'source_step':8000,'quota_boundary_step':7621,'uniform_origin_step':6500,'configured_end_step':25109,'Portuguese_per_update':1,'old_five_per_update':1,'prior_Portuguese_draws':100,'prior_old_five_draws':900}
        parent_config={'phase':ns['LOWER_LORA_PHASE'],'source_step':7750,'steps':25109,'config_identity':{'training_arguments':recipe,'domain_sampling_identity':{'parent':'samequota'}},
            'uniform_full_pass':{'quota_boundary':7621},'Portuguese_recovery':{'grace_new_updates':1000},'evaluation_languages':['tr','en','es','it','fr','de','pt']}
        parent_path=d/'parent.json';write_json(parent_path,parent_config);receipt_path=d/'receipt.json';write_json(receipt_path,{'chosen_step':8000,'completed_steps':8000,'artifacts':{'run_config':record(parent_path)}})
        declared={'source_step':8000,'quota_boundary_step':7621,'uniform_origin_step':6500,'configured_end_step':25109,'head_lr':1e-5,'lora_lr':1e-4,
            'all207_Adam_moments_and_global_clocks_preserved':True,'fresh_optimizer_or_sampler':False,'renewed_Portuguese_grace':False,'parent_failure_counter_preserved':True,'all_six_required_for_every_BEST':True,
            'declared_change':{'PT_replay_clips_per_update':1,'original_five_replay_clips_per_update':1,'PT_preservation_coefficients':[40,2,4],'original_five_preservation_coefficients':[40*(5/3),2*(5/3),4*(5/3)]}}
        gates={l:{'passed':l!='pt'} for l in ('en','es','it','fr','de','pt')}
        config={**parent_config,'phase':ns['PT_FOCUSED_PHASE'],'source_step':8000,'majority_source_step':7621,'uniform_origin_step':6500,'inherited_history_steps':8000,'remaining_uniform_pass_updates':17109,'trainable_parameter_tensors':207,'replay_clips_used':1914,
            'PT_replay_policy_transition':declared,'config_identity':{**parent_config['config_identity'],'domain_sampling_identity':{**parent_config['config_identity']['domain_sampling_identity'],'sampler_family':'TSC_uniform_majority_PT_focus_six_replay_sampler_v1','PT_focus_transition':focus},
                'PT_replay_transition':{'kind':'TSC_uniform_majority_PT_replay_state_transition_v1','declared':declared,'source_receipt':record(receipt_path),'parent_config_identity_sha256':ns['_mirror_config_sha'](parent_config['config_identity'])}},
            'initial_source_equivalence':{'passed':True,'actual_GPU_decode':True,'source_selected_step':8000,'compared_fixed_DEV_utterances':350,'all_raw_transcripts_and_emitted_paths_exact':True,'all_seven_ASR_counts_exact':True,
                'source_receipt':record(receipt_path),'source_retention_gates':gates,'all_six_retention_gates_passed':False}}
        base={'step':8000,'audio_seen_seconds':20000.,'turkish_audio_seen_seconds':14000.,'replay_audio_seen_seconds':6000.,'train_wall_seconds':1000.,'replay_language_clip_counts':prior_counts}
        # Old math is AST-identical and previously tested; stub only its context
        # admission/delegation here to isolate the new producer contract.
        ns['loralr_phase_contexts_for']=lambda c,h:{'legacy':'unchanged'}
        ns['loralr_phase_value_for_record']=lambda r,c,d:{'phase/majority_updates':r['step']-7621,'phase/lora_lr_fork_updates':r['step']-7750,'dev/inherited':1}
        contexts=ns['ptfocus_phase_contexts_for'](config,[base])
        row={**base,'step':8250,'PT_focus_updates':250,'PT_focus_source_step':8000,'audio_seen_seconds':25000.,'turkish_audio_seen_seconds':17500.,'replay_audio_seen_seconds':7500.,'train_wall_seconds':1100.,
            'replay_language_clip_counts':{**prior_counts,'pt':350,'en':430},'Turkish_domain_coverage':{'PT_focus_transition':{'source_step':8000,'local_updates':250,'Portuguese_draws_after_transition':250,'old_five_draws_after_transition':250}},
            'preservation_coefficients_by_language':{'pt':[40,2,4],'old_five':[40*5/3,2*5/3,4*5/3]}}
        values=ns['ptfocus_phase_value_for_record'](row,contexts,{})
        ok(values['resources/post_PT_focus_audio_minutes_per_wall_minute']==50 and values['resources/post_PT_focus_turkish_audio_minutes_per_wall_minute']==35 and values['phase/PT_focus_updates']==250,'Actual8000 phase rates/N separate from majority7621 andLR7750')
        ok(values['phase/PT_expected_KL_mean_density']==2.5 and math.isclose(values['phase/each_old_five_expected_KL_mean_density'],5/6) and values['phase/PT_expected_preservation_density_multiplier']==3,'ExpectedPTdensitytriples while eacholdfive densityretained; no gradientclaim')
        dev={8000:{'dev/wer_'+l+'_percent':40. for l in ('tr','en','es','it','fr','de','pt')}}
        initial=ns['ptfocus_phase_value_for_record'](base,contexts,dev)
        ok(initial['phase/PT_focus_updates']==0 and initial['retention/all_six_passed']==0 and initial['retention/Portuguese_stop_grace_active']==0,'Only actual8000 sevenDEV initial; noPTgrace/BEST fabrication')
        bad(lambda:ns['ptfocus_phase_value_for_record']({**row,'replay_language_clip_counts':{**row['replay_language_clip_counts'],'pt':349}},contexts,{}),'Wrong PT drawclock rejected')
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
        feature_receipt=d/'feature_receipt.json';write_json(feature_receipt,{'kind':'ROOT_owned_closed_TSC_uniform_retention_constraint_v1','chosen_step':8250,'completed_steps':8750,'chosen_checkpoint_role':'best','ROOT_review':record(measured_path),'artifacts':{'run_config':record(parent_integer)}})
        declaration={'source_step':8250,'failed_completed_step':8750,'source_feature_distillation':4.,'active_feature_distillation':20.,'rewound_TR_clips':7000,
            'ROOT_review':record(measured_path),'actual_diagnostic':None,'preservation_coefficients_by_language':{'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]},
            'all207_Adam_sampler_RNG_history_preserved':True,'original_integer_gate_correction_preserved':True,'new_counter_migration':False}
        constraint={'kind':'TSC_uniform_feature_retention_constraint_full_state_v1','source_step':8250,'source_receipt':record(feature_receipt),
            'parent_config_identity_sha256':ns['_mirror_config_sha'](integer_config['config_identity']),'declared':declaration,
            'fresh_optimizer_or_warmup':False,'Portuguese_grace_reset':False}
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
        bad(lambda:ns['feature_phase_contexts_for']({**feature_config,'retention_constraint_transition':{**constraint,'chosen_checkpoint_role':'latest'}},[base,row]),'Undeclared constraint envelope alteration rejected')
        ok(ns['feature_phase_baseline_steps'](feature_config)=={6500,7621,7750,8000,8250}
            and ns['feature_phase_initial_development_step'](feature_config)==8250,'Samebounded baselines/actualsource350 only')

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

        archive=d/'source_v16_shards';archive.mkdir();payload={'unknown':['retained','İ'],'finished':False,'reference_base_pt_verified_v1':{'sha256':'a'*64}}
        raw=json.dumps(payload).encode();digest=hashlib.sha256(raw).hexdigest();shard=archive/(digest+'.json');shard.write_bytes(raw)
        source_index=d/'source_v16.json';old_index={'schema':1,'kind':'Trackio_run_shard_index_v16','source_import':{'complete':True,'source_archive_directories':[str(archive)]},'runs':{'old':{'shard':record(shard),'finished_truthy':False}},'unknown_index_flag':{'exit1_review_preserved':True}}
        write_json(source_index,old_index);store=ns['StreamRunShardStore'](d/'new_v17.json');store.migrate(source_index,source_closed_confirmed=True)
        ok(store.index['source_import']['source_V16_index_payload']==old_index and store.activate('old')==payload,'Full V16 index/map/unknownflags imported unchanged')
        store.set_flag('new_flag',True);store.flush();ok(shard.exists() and read_json(source_index)==old_index,'Sourcearchives andsourceindex never rewritten/deleted')
    report={'schema':1,'kind':'inactive_Trackio_v17_PT_token_constraint_stdlib_verification_v1','passed':True,'checks':checks,'assertions':len(checks),'rejections':len(rejects),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'preserved_V16_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'legacy_functions_AST_unchanged':names,
        'scope':'Actual integer+feature+PTtoken context/value functions, synthetic hash-bound producer envelope/parentconfig/reviews and tiny actualshards. Old admission/value delegation explicitly stubbed; every oldmath AST unchanged. No actualmirror/history/DB/ML/GPU.'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'checks':len(checks),'source_sha256':report['source_sha256'],'proof_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'Trackio_v17_PT_token_constraint_stdlib_verification_v1.json');main(p.parse_args())
