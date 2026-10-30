"""Only new user-policy full-epoch boundary and byte-archive creator."""
from pathlib import Path
import ast
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish'
source=(base/'tsc_uniform_pttoken_constraint_transition_v1.py').read_text(encoding='utf-8')
source=source.replace('PT-token-only full-state fork from explicitly reviewed coherent latest8750','User-directed full-epoch continuation from coherent reviewed latest9000')
source=source.replace('Counter2','Counter3').replace('latest8750','latest9000').replace('own8750','own9000')
source=source.replace('import tsc_uniform_retention_constraint_transition_v1 as parent','import tsc_uniform_pttoken_constraint_transition_v1 as parent')
source=source.replace('ROOT_owned_closed_TSC_uniform_PT_token_constraint_v1','ROOT_owned_closed_TSC_uniform_full_epoch_v1')
source=source.replace('ROOT_delegated_measured_TSC_PT_token_constraint_review_v1','ROOT_USER_TSC_full_epoch_policy_review_v1')
start=source.index('PARENT_WORKER=');end=source.index('\n\ndef token_constraint_declaration',start)
source=source[:start]+'''PARENT_WORKER='train_lora_batched_tsc_uniform_pttoken_v1.py'
PARENT_CONTROLLER='run_TSC_uniform_pttoken_v1.py'
PARENT_PINS={PARENT_WORKER:'41c15b88156a449c81ab5c8f6049812d77721bf410b6cd3d30d29978d3e479a0',
    PARENT_CONTROLLER:'5af40e6f640c66c3136473cc29fb30013ae854a5daa2c091837b5f1d5ce2e066',
    'tsc_uniform_pttoken_constraint_transition_v1.py':'e5afc8f1a639893e8c5460e724fc3557e9d21765594286ea89b8337dbd64acfb'}
'''+source[end:]
start=source.index('def token_constraint_declaration');end=source.index('\n\ndef ptfocus_declaration',start)
source=source[:start]+'''def full_epoch_declaration(context):
    return {'source_portuguese_token_kl_multiplier':5.,'active_portuguese_token_kl_multiplier':1.,
        'PT_token_KL_before':200.,'PT_token_KL_after':40.,'base_token_KL':40.,'duration_KL':2.,'feature_distillation':20.,
        'source_step':9000,'preservation_coefficients_by_language':{'pt':[40.,2.,20.],'old_five':[40*(5/3),2*(5/3),20*(5/3)]},
        'source_retention_stop_policy':'enforced','active_retention_stop_policy':'monitor_only',
        'retention_gates_for_champion_selection_unchanged':True,'retention_failure_counters_and_history_preserved':True,
        'retention_early_stops_disabled_before_actual_full_unique_pass':True,
        'resource_and_nonfinite_protections_unchanged':True,'all207_Adam_sampler_RNG_history_preserved':True,
        'ROOT_review':context['receipt']['ROOT_review'],'renewed_Portuguese_grace':False,
        'source_actual_retention_failures':3,'source_actual_selected_step':8250,'new_counter_migration':False,
        'rewound_TR_clips':0,'no_hybrid_training_state':True,'fresh_optimizer_or_warmup':False,
        'source_chosen_is_all_six_qualified':False,'full_pass_requires_actual_unique_TR_clips':253798}
'''+source[end:]
source=source.replace('actual_CLOSED_reviewed_latest_PT_token_constraint','actual_CLOSED_USER_full_epoch_monitor_only')
source=source.replace("==8750", "==9000").replace("'chosen_step')==value.get('completed_steps')==8750", "'chosen_step')==value.get('completed_steps')==9000")
source=source.replace("value.get('source_counter_preserved')==2", "value.get('source_counter_preserved')==3")
source=source.replace('allow_declared_PT_token_constraint_fork','allow_full_epoch_monitor_only_continuation')
source=source.replace("review.get('source_counter_preserved')==2", "review.get('source_counter_preserved')==3")
source=source.replace("and review.get('first_future_complete_DEV_must_pass_all_six_or_stop_counter3') is True", "and review.get('retention_stop_policy')=='monitor_only'")
start=source.index("    change=review['declared_change'];");end=source.index("    closed=read",start)
source=source[:start]+'''    change=review['declared_change'];require(change['parameter']=='portuguese_token_kl_multiplier'
        and change['source_value']==5 and change['active_value']==1 and change['PT_token_KL_before']==200
        and change['PT_token_KL_after']==40 and change['baseKD']==40 and change['old_five_token_KL_unchanged']==40*(5/3)
        and change['feature_distillation_unchanged']==20 and change['duration_KL_unchanged']==2
        and change['head_LR_unchanged']==1e-5 and change['LoRA_LR_unchanged']==1e-4
        and change['TR_per_update']==14 and change['PT_replay_per_update']==change['old_five_replay_per_update']==1
        and change['fixed_integer_gate_limits_unchanged'] is True,
        'Only PT token5 to1 rollback and explicit full-epoch retention-stop policy may change')
'''+source[end:]
source=source.replace(";checked_record(review['actual_diagnostic'])",'')
source=source.replace("{'session_id':98758,'exit_code':0,'tool_chunk':'0848c0'}", "{'session_id':37834,'exit_code':0,'tool_chunk':'af6052'}")
source=source.replace("selection['consecutive_retention_failures']==2", "selection['consecutive_retention_failures']==3")
source=source.replace("_history_frontier(a['training_history']['path'],8750,8750)", "_history_frontier(a['training_history']['path'],9000,9000)")
source=source.replace('original_feature_source_receipt','original_PT_token_source_receipt')
source=source.replace("and recipe['lora_lr']==1e-4,", "and recipe['lora_lr']==1e-4 and recipe['portuguese_token_kl_multiplier']==5,")
start=source.index("    require(all(gates[lang]['passed']");end=source.index("    require(last['validation']",start)
source=source[:start]+'''    require(not gates['de']['passed'] and not gates['pt']['passed'] and gates['de']['word_errors']==71
        and gates['de']['allowed_word_errors']==70 and gates['pt']['word_errors']==71 and gates['pt']['allowed_word_errors']==66
        and all(v['passed'] for v in best_gates.values())
        and best_summary['validation_tr']['asr']['wer']==selection['turkish_validation_wer'],
        'Reviewed latestDE71/PT71 failure and separate all-six BEST8250 must remain truthful')
'''+source[end:]
source=source.replace("last['selection_diagnostics']['consecutive_retention_failures']==2", "last['selection_diagnostics']['consecutive_retention_failures']==3")
source=source.replace('==24774','==28274')
source=source.replace("'pttoken_review':review", "'full_epoch_review':review")
source=source.replace("'source_recipe':{**recipe,'portuguese_token_kl_multiplier':5.}", "'source_recipe':{**recipe,'portuguese_token_kl_multiplier':1.,'retention_stop_policy':'monitor_only'}")
source=source.replace("'source_step':8750,'source_selected_step':8750", "'source_step':9000,'source_selected_step':9000")
source=source.replace("'source_actual_failure_counter':2,'corrected_initial_retention_failures':2", "'source_actual_failure_counter':3,'corrected_initial_retention_failures':3")
source=source.replace("'prior_uniform_draws':24774,'remaining_updates_from_chosen_frontier':16359", "'prior_uniform_draws':28274,'remaining_updates_from_chosen_frontier':16109")
source=source.replace("state['retention_failures']==2", "state['retention_failures']==3")
source=source.replace("len(state['history'])==8750", "len(state['history'])==9000")
source=source.replace("'source_actual_failure_counter':2,'Portuguese_token_multiplier_change_only':[1.,5.]", "'source_actual_failure_counter':3,'Portuguese_token_multiplier_change_only':[5.,1.],\n        'retention_stop_policy':'monitor_only'")
source=source.replace("PT_token_constraint_and_original_source_caches_cleared", "full_epoch_policy_and_original_source_caches_cleared")
source=source.replace('counter2','counter3').replace('Latest8750','Latest9000').replace('Actual delegated latest8750','Actual user-directed latest9000')
helper=base/'tsc_uniform_full_epoch_transition_v1.py';assert not helper.exists();ast.parse(source);helper.write_text(source,encoding='utf-8')

creator=(base/'record_TSC_uniform_pttoken_constraint_source_v1.py').read_text(encoding='utf-8')
creator=creator.replace('tsc_uniform_pttoken_constraint_transition_v1','tsc_uniform_full_epoch_transition_v1')
creator=creator.replace('PT-token-only','user full-epoch').replace('latest8750','latest9000').replace('latest8750/counter2','latest9000/counter3')
creator=creator.replace('allow_declared_PT_token_constraint_fork','allow_full_epoch_monitor_only_continuation')
creator=creator.replace("==8750", "==9000").replace("==2", "==3")
creator=creator.replace(";checked_record(review['actual_diagnostic'])",'')
creator=creator.replace("_history_frontier(a['training_history']['path'],8750,8750)", "_history_frontier(a['training_history']['path'],9000,9000)")
creator=creator.replace('actual_CLOSED_reviewed_latest_PT_token_constraint','actual_CLOSED_USER_full_epoch_monitor_only')
creator=creator.replace("'chosen_step':8750,'completed_steps':8750", "'chosen_step':9000,'completed_steps':9000")
creator=creator.replace('original_feature_source_receipt','original_PT_token_source_receipt')
creator=creator.replace("'source_counter_preserved':2", "'source_counter_preserved':3")
creator=creator.replace("'source_step':8750", "'source_step':9000")
creator=creator.replace("'active_PT_token_multiplier':5", "'active_PT_token_multiplier':1,'retention_stop_policy':'monitor_only'")
creator=creator.replace("'failed_branch_history_preserved':True", "'failed_branch_history_preserved':True,'retention_stop_policy':'monitor_only'")
file=base/'record_TSC_uniform_full_epoch_source_v1.py';assert not file.exists();ast.parse(creator);file.write_text(creator,encoding='utf-8')
