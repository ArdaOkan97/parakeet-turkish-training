"""New technical IO source boundary/creator, preserving all old sources."""
from pathlib import Path
import ast
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish'
source=(base/'tsc_uniform_full_epoch_transition_v1.py').read_text(encoding='utf-8')
source=source.replace('User-directed full-epoch continuation from coherent reviewed latest9000','Matched100-update IO control from coherent CLOSED latest25109')
source=source.replace('Counter3 and separate qualified BEST8250','Counter23 and separate qualified BEST19750')
source=source.replace('import tsc_uniform_pttoken_constraint_transition_v1 as parent','import tsc_uniform_full_epoch_transition_v1 as parent')
source=source.replace('ROOT_owned_closed_TSC_uniform_full_epoch_v1','ROOT_owned_closed_TSC_uniform_IO_control_v1')
source=source.replace('ROOT_USER_TSC_full_epoch_policy_review_v1','ROOT_USER_TSC_uniform_IO_control_review_v1')
start=source.index('PARENT_WORKER=');end=source.index('\n\ndef full_epoch_declaration',start)
source=source[:start]+'''PARENT_WORKER='train_lora_batched_tsc_uniform_full_epoch_v1.py'
PARENT_CONTROLLER='run_TSC_uniform_full_epoch_v1.py'
PARENT_PINS={PARENT_WORKER:'1c33cbde6933c55d0839d0ebf82253fb3b673dde7f5a105adc3905e14f7013c3',
    PARENT_CONTROLLER:'1943266496b654bc3439c0f667f8a197bb620a0c1d110d5bd9269cb26910d885',
    'tsc_uniform_full_epoch_transition_v1.py':'7cf6aaa78ef3372353cd3320aa606add08927dbecca17a3e91dfd482474b3ce3'}
PERSISTENCE_MODES=('double_json','single_serialize_two_commits')
'''+source[end:]
start=source.index('def full_epoch_declaration');end=source.index('\n\ndef ptfocus_declaration',start)
source=source[:start]+'''def io_declaration(context,persistence_mode):
    require(persistence_mode in PERSISTENCE_MODES and persistence_mode==context['receipt']['persistence_mode'],
        'Exact reviewed persistence mode required')
    return {'persistence_mode':persistence_mode,'source_step':25109,'technical_updates':100,'technical_end_step':25209,
        'source_lineage_end_step':25109,'source_actual_retention_failures':23,'source_actual_selected_step':19750,
        'source_uniform_unique_clips':253798,'source_uniform_draws':253800,'source_repeat_visits':2,
        'all207_weights_Adam_sampler_RNG_history_preserved':True,'training_math_LRs_teacher_and_inputs_unchanged':True,
        'paired_publication_frequency_schema_and_commit_order_unchanged':True,'strict_all_six_champion_gates_unchanged':True,
        'resource_nonfinite_protections_unchanged':True,'retention_stop_policy':'monitor_only',
        'ROOT_review':context['receipt']['ROOT_review'],'coverage_rewind_clips':0,'sampler_queues_or_RNG_reset':False,
        'technical_test_is_quality_recipe_screen_or_new_full_pass':False}
'''+source[end:]
source=source.replace('actual_CLOSED_USER_full_epoch_monitor_only','actual_CLOSED_matched_IO_control')
source=source.replace("==9000", "==25109").replace("==8250", "==19750")
source=source.replace("value.get('source_counter_preserved')==3", "value.get('source_counter_preserved')==23")
source=source.replace("and value.get('explicit_coverage_rewind_clips')==0 and value.get('source_counter_preserved')==23,", "and value.get('explicit_coverage_rewind_clips')==0 and value.get('source_counter_preserved')==23\n        and value.get('persistence_mode') in PERSISTENCE_MODES,")
source=source.replace('allow_full_epoch_monitor_only_continuation','allow_matched_IO_control')
source=source.replace("review.get('source_counter_preserved')==3", "review.get('source_counter_preserved')==23")
source=source.replace("and review['declared_change'].get('active_retention_stop_policy')=='monitor_only'", "and review.get('technical_updates')==100 and review.get('technical_end_step')==25209")
start=source.index("    change=review['declared_change'];");end=source.index("    closed=read",start)
source=source[:start]+'''    require(review.get('persistence_modes')==list(PERSISTENCE_MODES)
        and review.get('training_math_LRs_teacher_and_inputs_unchanged') is True
        and review.get('paired_publication_frequency_schema_and_commit_order_unchanged') is True,
        'Only matched persistence implementation and explicit100-update execution extension may change')
'''+source[end:]
source=source.replace("{'session_id':37834,'exit_code':0,'tool_chunk':'af6052'}", "{'session_id':67197,'exit_code':0,'tool_chunk':'90d47e'}")
source=source.replace("selection['consecutive_retention_failures']==3", "selection['consecutive_retention_failures']==23")
source=source.replace("_history_frontier(a['training_history']['path'],9000,9000)", "_history_frontier(a['training_history']['path'],25109,25109)")
source=source.replace('original_PT_token_source_receipt','original_full_epoch_source_receipt')
source=source.replace("recipe['portuguese_token_kl_multiplier']==5,", "recipe['portuguese_token_kl_multiplier']==1 and recipe['retention_stop_policy']=='monitor_only' and recipe['steps']==25109,")
start=source.index("    require(not gates['de']['passed']");end=source.index("    require(last['validation']",start)
source=source[:start]+'''    require(not gates['es']['passed'] and gates['es']['word_errors']==53 and gates['es']['allowed_word_errors']==52
        and all(gates[lang]['passed'] for lang in ('en','it','fr','de','pt'))
        and all(v['passed'] for v in best_gates.values())
        and best_summary['validation_tr']['asr']['wer']==selection['turkish_validation_wer'],
        'Unqualified latest25109 and separate all-six BEST19750 must remain truthful')
'''+source[end:]
source=source.replace("last['selection_diagnostics']['consecutive_retention_failures']==3", "last['selection_diagnostics']['consecutive_retention_failures']==23")
source=source.replace("['unique_clips_seen']==28274", "['unique_clips_seen']==253798")
source=source.replace("['draws']==28274", "['draws']==253800\n        and selection['Turkish_domain_coverage']['uniform_pass']['repeat_visits']==2\n        and selection['Turkish_domain_coverage']['uniform_pass']['full_pass_complete'] is True")
source=source.replace("'full_epoch_review':review", "'io_review':review")
source=source.replace("'source_recipe':{**recipe,'portuguese_token_kl_multiplier':1.,'retention_stop_policy':'monitor_only'}", "'source_recipe':{**recipe,'steps':25209,'persistence_mode':receipt['persistence_mode']}")
source=source.replace("'source_step':9000,'source_selected_step':9000", "'source_step':25109,'source_selected_step':25109")
source=source.replace("'source_actual_selected_step':8250,'source_actual_failure_counter':3,'corrected_initial_retention_failures':3", "'source_actual_selected_step':19750,'source_actual_failure_counter':23,'corrected_initial_retention_failures':23")
source=source.replace("'corrected_initial_selected_step':8250", "'corrected_initial_selected_step':19750")
source=source.replace("'source_best_step':8250", "'source_best_step':19750")
source=source.replace("'prior_uniform_draws':28274,'remaining_updates_from_chosen_frontier':16109", "'prior_uniform_draws':253800,'remaining_updates_from_chosen_frontier':100,\n        'source_lineage_end_step':25109,'technical_end_step':25209,'technical_updates':100")
source=source.replace("'configured_end_step':25109,'uniform_origin_step'", "'configured_end_step':25209,'uniform_origin_step'")
source=source.replace("state['retention_failures']==3", "state['retention_failures']==23")
source=source.replace("len(state['history'])==9000", "len(state['history'])==25109")
source=source.replace("'source_actual_selected_step':8250,'source_actual_failure_counter':3,'Portuguese_token_multiplier_change_only':[5.,1.],", "'source_actual_selected_step':19750,'source_actual_failure_counter':23,'training_math_LRs_unchanged':True,")
source=source.replace('counter3','counter23').replace('latest9000','latest25109').replace('Latest9000','Latest25109').replace('BEST8250','BEST19750').replace('own9000','own25109')
source=source.replace('full_epoch_policy_and_original_source_caches_cleared','IO_control_and_original_source_caches_cleared')
file=base/'tsc_uniform_io_transition_v1.py';assert not file.exists();ast.parse(source);file.write_text(source,encoding='utf-8')

creator=(base/'record_TSC_uniform_full_epoch_source_v1.py').read_text(encoding='utf-8')
creator=creator.replace('tsc_uniform_full_epoch_transition_v1','tsc_uniform_io_transition_v1')
creator=creator.replace('validate_receipt,source_context)', 'validate_receipt,source_context,PERSISTENCE_MODES)')
creator=creator.replace('allow_full_epoch_monitor_only_continuation','allow_matched_IO_control')
creator=creator.replace('==9000','==25109').replace('==8250','==19750').replace("==3", "==23")
creator=creator.replace("_history_frontier(a['training_history']['path'],9000,9000)", "_history_frontier(a['training_history']['path'],25109,25109)")
creator=creator.replace('actual_CLOSED_USER_full_epoch_monitor_only','actual_CLOSED_matched_IO_control')
creator=creator.replace("'chosen_step':9000,'completed_steps':9000", "'chosen_step':25109,'completed_steps':25109")
creator=creator.replace('original_PT_token_source_receipt','original_full_epoch_source_receipt')
creator=creator.replace("'source_counter_preserved':3", "'source_counter_preserved':23,'persistence_mode':args.persistence_mode")
creator=creator.replace("'source_step':9000", "'source_step':25109")
creator=creator.replace("'separate_selected_BEST_step':8250", "'separate_selected_BEST_step':19750")
creator=creator.replace("'active_PT_token_multiplier':1,'retention_stop_policy':'monitor_only'", "'persistence_mode':args.persistence_mode,'technical_end_step':25209")
creator=creator.replace("p.add_argument('--target-run-id',required=True);", "p.add_argument('--target-run-id',required=True);p.add_argument('--persistence-mode',choices=PERSISTENCE_MODES,required=True);")
creator=creator.replace('latest9000','latest25109').replace('counter3','counter23').replace('BEST8250','BEST19750').replace('user full-epoch','matched technical IO')
file=base/'record_TSC_uniform_io_source_v1.py';assert not file.exists();ast.parse(creator);file.write_text(creator,encoding='utf-8')
