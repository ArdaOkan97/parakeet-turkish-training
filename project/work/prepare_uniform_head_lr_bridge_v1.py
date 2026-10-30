"""Create future inactive head-rate source contract; no admission execution."""
from pathlib import Path
import ast
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish'
source=(base/'tsc_uniform_encoder_schedule_transition_v1.py').read_text(encoding='utf-8')
source=source.replace('Matched encoder-only full-shuffle-cycle schedule control','Inactive future matched15-head learning-rate control')
source=source.replace('Counter23 and separate qualified BEST19750 are carried unchanged.',
    'Actual encoder-policy selection and a future source receipt are required.\nCounter23 and separate qualified BEST19750 are carried unchanged.')
source=source.replace('ROOT_owned_closed_TSC_uniform_encoder_schedule_v1','ROOT_owned_closed_TSC_uniform_head_LR_v1')
source=source.replace('ROOT_USER_TSC_uniform_encoder_schedule_review_v1','ROOT_USER_TSC_uniform_head_LR_control_review_v1')
source=source.replace("ENCODER_SCHEDULE_MODES=('constant','cosine')", "ENCODER_SCHEDULE_MODES=('constant','cosine')\nHEAD_LR_CHOICES=(1e-5,3e-5)\nPOLICY_CHOICE_KIND='ROOT_completed_encoder_schedule_policy_selection_v1'")
source=source.replace("'shared_head_lr_unchanged':1e-5,", "'source_shared_head_lr':1e-5,'active_shared_head_lr':context['receipt']['head_lr'],\n        'shared_head_lr_change_is_separately_declared':True,")
start=source.index('\n\ndef ptfocus_declaration')
source=source[:start]+'''

def head_lr_declaration(context,encoder_lr_schedule,head_lr):
    require(head_lr in HEAD_LR_CHOICES and head_lr==context['receipt']['head_lr'], 'Only bound1e-5 or3e-5 head rate admitted')
    encoder=encoder_schedule_declaration(context,encoder_lr_schedule)
    return {'source_head_lr':1e-5,'active_head_lr':head_lr,'source_step':25109,
        'selected_encoder_policy':encoder_lr_schedule,'encoder_policy_choice_review':context['head_lr_review']['encoder_policy_choice_review'],
        'encoder_schedule':encoder,'all15_head_tensors_already_active':True,'head_tensors_newly_unfrozen':0,
        'affected_head_modules':['encoder_projector','decoder.embedding','decoder.lstm','decoder.decoder_projector','joint.head'],
        'six_glyph_post_Adam_displacement_multiplier_unchanged':6.,
        'old_lexical_blank_and_duration_parameters_remain_active':True,
        'source_head_Adam_moments_clocks_and_decay_preserved':True,'all207_state_sampler_RNG_history_preserved':True,
        'only_head_group_rate_changes_relative_to_selected_encoder_policy':True,
        'matched_updates':18129,'matched_end_step':43238,'ROOT_review':context['receipt']['ROOT_review'],
        'source_actual_retention_failures':23,'source_actual_selected_step':19750,
        'coverage_rewind_clips':0,'renewed_Portuguese_grace':False,'fresh_optimizer_or_warmup':False}
'''+source[start:]
source=source.replace('actual_CLOSED_matched_encoder_schedule','actual_CLOSED_selected_encoder_policy_head_LR_control')
source=source.replace("and value.get('encoder_lr_schedule') in ENCODER_SCHEDULE_MODES,", "and value.get('encoder_lr_schedule') in ENCODER_SCHEDULE_MODES and value.get('head_lr') in HEAD_LR_CHOICES,")
source=source.replace('allow_matched_encoder_schedule_control','allow_matched_head_LR_control')
needle="    change=review['declared_change']"
idx=source.index(needle)
source=source[:idx]+'''    require(review.get('encoder_policy_selection_completed') is True
        and review.get('selected_encoder_policy')==receipt['encoder_lr_schedule'],
        'Actual reviewed selected encoder policy is required; no choice inferred from preparation')
    policy=read(checked_record(review['encoder_policy_choice_review']))
    require(policy.get('kind')==POLICY_CHOICE_KIND and policy.get('passed') is True
        and policy.get('actual_curve_evidence_used') is True
        and policy.get('selected_encoder_lr_schedule')==receipt['encoder_lr_schedule'],
        'Completed actual encoder-curve selection evidence required before source admission')
'''+source[idx:]
start=source.index("    require(change.get('parameter')");end=source.index("    closed=read",start)
source=source[:start]+'''    require(change.get('parameter')=='shared_head_lr' and change.get('choices')==list(HEAD_LR_CHOICES)
        and change.get('source_head_lr')==1e-5 and change.get('active_head_lr')==receipt['head_lr']
        and change.get('selected_encoder_policy')==receipt['encoder_lr_schedule']
        and change.get('additional_updates')==18129 and change.get('end_step')==43238
        and change.get('source_encoder_lr')==1e-4 and change.get('minimum_encoder_lr')==3e-5
        and change.get('legacy_schedule')=='constant' and change.get('warmup_reset') is False
        and change.get('persistence_mode')=='double_json' and change.get('all15_head_tensors_already_active') is True
        and change.get('six_glyph_post_Adam_displacement_multiplier_unchanged')==6
        and change.get('all_other_recipe_fields_unchanged') is True and change.get('no_coverage_rewind') is True
        and change.get('strict_all_six_winner_gates_unchanged') is True and change.get('retention_stop_policy')=='monitor_only',
        'Only head-group rate relative to explicitly selected encoder policy may change')
'''+source[end:]
source=source.replace("'encoder_schedule_review':review", "'head_lr_review':review")
source=source.replace("'source_recipe':{**recipe,'steps':43238,'persistence_mode':'double_json',", "'source_recipe':{**recipe,'steps':43238,'persistence_mode':'double_json','head_lr':receipt['head_lr'],")
source=source.replace('encoder_schedule_and_original_source_caches_cleared','head_LR_and_original_source_caches_cleared')
helper=base/'tsc_uniform_head_lr_transition_v1.py';assert not helper.exists();ast.parse(source);helper.write_text(source,encoding='utf-8')

creator=(base/'record_TSC_uniform_encoder_schedule_source_v1.py').read_text(encoding='utf-8')
creator=creator.replace('tsc_uniform_encoder_schedule_transition_v1','tsc_uniform_head_lr_transition_v1')
creator=creator.replace('source_context,ENCODER_SCHEDULE_MODES)', 'source_context,ENCODER_SCHEDULE_MODES,HEAD_LR_CHOICES,POLICY_CHOICE_KIND)')
creator=creator.replace('allow_matched_encoder_schedule_control','allow_matched_head_LR_control')
idx=creator.index("    closed=read(checked_record(review['actual_closed_source_review']))")
creator=creator[:idx]+'''    require(review.get('encoder_policy_selection_completed') is True
        and review.get('selected_encoder_policy')==args.encoder_lr_schedule,
        'Explicit actual encoder-policy selection required before any checkpoint bytes')
    policy=read(checked_record(review['encoder_policy_choice_review']))
    require(policy.get('kind')==POLICY_CHOICE_KIND and policy.get('passed') is True
        and policy.get('actual_curve_evidence_used') is True
        and policy.get('selected_encoder_lr_schedule')==args.encoder_lr_schedule,
        'Actual encoder-curve evidence must choose the future policy')
'''+creator[idx:]
creator=creator.replace('actual_CLOSED_matched_encoder_schedule','actual_CLOSED_selected_encoder_policy_head_LR_control')
creator=creator.replace("'encoder_lr_schedule':args.encoder_lr_schedule,", "'encoder_lr_schedule':args.encoder_lr_schedule,'head_lr':args.head_lr,")
creator=creator.replace("p.add_argument('--encoder-lr-schedule',choices=ENCODER_SCHEDULE_MODES,required=True);", "p.add_argument('--encoder-lr-schedule',choices=ENCODER_SCHEDULE_MODES,required=True);p.add_argument('--head-lr',type=float,choices=HEAD_LR_CHOICES,required=True);")
creator=creator.replace('matched encoder schedule', 'inactive future matched head-rate')
file=base/'record_TSC_uniform_head_lr_source_v1.py';assert not file.exists();ast.parse(creator);file.write_text(creator,encoding='utf-8')
