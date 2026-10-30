"""Create only new inactive declared LR boundary/creator siblings."""
from pathlib import Path
import ast
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish'
source=(base/'tsc_uniform_retention_constraint_transition_v1.py').read_text(encoding='utf-8')
source=source.replace('Measured one-variable feature-KD fork','Measured one-variable encoder-LR fork')
source=source.replace('explicit7000-clip rewind','explicit10500-clip rewind from failed9000')
source=source.replace('import tsc_uniform_majority_ptfocus_integer_transition_v1 as parent','import tsc_uniform_retention_constraint_transition_v1 as parent')
source=source.replace('ROOT_owned_closed_TSC_uniform_retention_constraint_v1','ROOT_owned_closed_TSC_uniform_feature_LoRA_LR_v1')
source=source.replace('ROOT_delegated_measured_TSC_retention_constraint_review_v1','ROOT_delegated_measured_TSC_feature_LoRA_LR_review_v1')
start=source.index("PARENT_WORKER=");end=source.index('\n\ndef constraint_declaration',start)
source=source[:start]+'''PARENT_WORKER='train_lora_batched_tsc_uniform_constraint_v2.py'
PARENT_CONTROLLER='run_TSC_uniform_constraint_v2.py'
PARENT_PINS={PARENT_WORKER:'f6433bd9cf34a1560f0a0093630659ef365f151e8bad68f6bfc4b9357b7731fe',
    PARENT_CONTROLLER:'35bd6d564f91ecc2cc1db3d3b4cac59ef0e40a6f4b27c19d122b5fecf4db9d5c',
    'tsc_uniform_retention_constraint_transition_v1.py':'ccdef7862d6fec68d28265199514ebdaa4772352be47cc4cfec0230e8370bfba'}
'''+source[end:]
start=source.index('def constraint_declaration');end=source.index('\n\ndef ptfocus_declaration',start)
source=source[:start]+'''def lora_lr_declaration(context):
    return {'source_lora_lr':1e-4,'active_lora_lr':3e-5,'source_step':8250,
        'feature_distillation_unchanged':20.,'token_KL_unchanged':40.,'duration_KL_unchanged':2.,'head_LR_unchanged':1e-5,
        'failed_completed_step':9000,'rewound_TR_clips':10500,'chosen_uniform_unique_clips':17774,
        'preservation_coefficients_by_language':{'pt':[40.,2.,20.],'old_five':[40*(5/3),2*(5/3),20*(5/3)]},
        'all207_Adam_sampler_RNG_history_preserved':True,'ROOT_review':context['receipt']['ROOT_review'],
        'actual_diagnostic':context['lora_lr_review']['actual_diagnostic'],
        'recent_failed_PT_token_review':context['lora_lr_review']['recent_failed_PT_token_review'],
        'original_integer_gate_correction_preserved':True,'renewed_Portuguese_grace':False,
        'source_actual_retention_failures':0,'source_actual_selected_step':8250,'new_counter_migration':False,
        'no_hybrid_training_state':True,'source_Adam_moments_or_clock_reset':False}


def constraint_declaration(context):return context['config']['retention_constraint_transition']['declared']
'''+source[end:]
source=source.replace('actual_CLOSED_qualified_BEST_measured_feature_constraint','actual_CLOSED_qualified_BEST_measured_lower_LoRA')
source=source.replace("value.get('explicit_coverage_rewind_clips')==7000", "value.get('explicit_coverage_rewind_clips')==10500")
source=source.replace('allow_declared_feature_distillation_constraint_fork','allow_declared_feature_LoRA_LR_fork')
source=source.replace("review.get('explicit_coverage_rewind_clips')==7000", "review.get('explicit_coverage_rewind_clips')==10500")
start=source.index("    change=review['declared_change'];");end=source.index("    closed=read",start)
source=source[:start]+'''    change=review['declared_change'];require(change['parameter']=='lora_lr' and change['source_value']==1e-4
        and change['active_value']==3e-5 and change['feature_distillation_unchanged']==20
        and change['token_KL_unchanged']==40 and change['duration_KL_unchanged']==2 and change['head_LR_unchanged']==1e-5
        and change['TR_per_update']==14 and change['PT_replay_per_update']==change['old_five_replay_per_update']==1
        and change['old_five_language_weight_multiplier']==5/3 and change['fixed_integer_gate_limits_unchanged'] is True,
        'Only active encoder LoRA LR1e-4 to3e-5 may change')
'''+source[end:]
source=source.replace("require(review['source_actual_owned_exit']['exit_code']==0 and review['actual_diagnostic_owned_exit']['exit_code']==0", "require(review['source_actual_owned_exit']=={'session_id':98758,'exit_code':0,'tool_chunk':'0848c0'}")
needle="    for key,item in a.items():checked_record"
idx=source.index(needle)
source=source[:idx]+'''    failed=read(checked_record(review['recent_failed_PT_token_review']))
    require(failed['completed_steps']==9000 and review['failed_branch_history_preserved'] is True
        and review['chosen_uniform_unique_clips']==17774 and review['failed_uniform_unique_clips']==28274,
        'Failed9000 history and explicit10500-clip rewind remain bound')
'''+source[idx:]
source=source.replace('original_integer_source_receipt','original_feature_source_receipt')
source=source.replace("recipe['feature_distillation']==4", "recipe['feature_distillation']==20 and recipe['lora_lr']==1e-4")
source=source.replace('Original source recipe must be feature4 and unchanged otherwise','Actual source feature20/LoRA1e-4 must be unchanged otherwise')
source=source.replace("config['retention_numeric_correction']['ROOT_review']==origin['receipt']['ROOT_review']", "config['retention_numeric_correction']==origin['parent_identity']['retention_numeric_correction']")
source=source.replace("'constraint_review':review", "'lora_lr_review':review")
source=source.replace("'source_recipe':{**recipe,'feature_distillation':20.}", "'source_recipe':{**recipe,'lora_lr':3e-5}")
source=source.replace("'combined_baseline':baseline,'failed_completed_step':8750", "'combined_baseline':baseline,'failed_completed_step':9000,\n        'prior_uniform_draws':17774,'remaining_updates_from_chosen_frontier':16859,'source_best_summary':summary,\n        'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,'focus_origin_step':8000")
source=source.replace("'feature_coefficient_change_only':[4.,20.]", "'active_encoder_LR_change_only':[1e-4,3e-5]")
source=source.replace("'hybrid_or_latest8750_state_used':False", "'hybrid_or_latest_state_used':False")
source=source.replace('feature_constraint_and_original_source_caches_cleared','feature_LoRA_LR_and_original_source_caches_cleared')
helper=base/'tsc_uniform_feature_loralr_transition_v1.py';assert not helper.exists();ast.parse(source);helper.write_text(source,encoding='utf-8')

creator=(base/'record_TSC_uniform_retention_constraint_source_v1.py').read_text(encoding='utf-8')
creator=creator.replace('feature-only','encoder-LR-only').replace('tsc_uniform_retention_constraint_transition_v1','tsc_uniform_feature_loralr_transition_v1')
creator=creator.replace('allow_declared_feature_distillation_constraint_fork','allow_declared_feature_LoRA_LR_fork')
creator=creator.replace("and review['actual_diagnostic_owned_exit']['exit_code']==0",'')
creator=creator.replace('actual_CLOSED_qualified_BEST_measured_feature_constraint','actual_CLOSED_qualified_BEST_measured_lower_LoRA')
creator=creator.replace("'explicit_coverage_rewind_clips':7000", "'explicit_coverage_rewind_clips':10500")
creator=creator.replace("'explicit_rewind_TR_clips':7000,'active_feature_coefficient':20", "'explicit_rewind_TR_clips':10500,'active_LoRA_LR':3e-5")
creator=creator.replace('original_integer_source_receipt','original_feature_source_receipt')
creator=creator.replace('full8750 history remain separate from chosen8250','full8750 and failed9000 histories remain separate from chosen8250')
creator=creator.replace("'failed_latest_step':8750", "'failed_latest_step':9000")
file=base/'record_TSC_uniform_feature_loralr_source_v1.py';assert not file.exists();ast.parse(creator);file.write_text(creator,encoding='utf-8')
