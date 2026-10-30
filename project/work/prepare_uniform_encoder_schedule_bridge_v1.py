"""Create only new schedule-boundary and original-full-source archive creator."""
from pathlib import Path
import ast
root=Path(__file__).resolve().parent.parent;base=root/'outputs/parakeet_turkish'
source=(base/'tsc_uniform_io_transition_v1.py').read_text(encoding='utf-8')
source=source.replace('Matched100-update IO control','Matched encoder-only full-shuffle-cycle schedule control')
source=source.replace('ROOT_owned_closed_TSC_uniform_IO_control_v1','ROOT_owned_closed_TSC_uniform_encoder_schedule_v1')
source=source.replace('ROOT_USER_TSC_uniform_IO_control_review_v1','ROOT_USER_TSC_uniform_encoder_schedule_review_v1')
source=source.replace("PERSISTENCE_MODES=('double_json','single_serialize_two_commits')", "ENCODER_SCHEDULE_MODES=('constant','cosine')")
start=source.index('def io_declaration');end=source.index('\n\ndef ptfocus_declaration',start)
source=source[:start]+'''def encoder_schedule_declaration(context,encoder_lr_schedule):
    require(encoder_lr_schedule in ENCODER_SCHEDULE_MODES and encoder_lr_schedule==context['receipt']['encoder_lr_schedule'],
        'Exact reviewed encoder schedule mode required')
    return {'encoder_lr_schedule':encoder_lr_schedule,'source_step':25109,'encoder_schedule_updates':18129,
        'encoder_schedule_end_step':43238,'source_lineage_end_step':25109,
        'source_encoder_lr':1e-4,'first_applied_encoder_lr':1e-4,'encoder_min_lr':3e-5,
        'last_applied_encoder_lr':3e-5 if encoder_lr_schedule=='cosine' else 1e-4,
        'shared_head_lr_unchanged':1e-5,'legacy_schedule_unchanged':'constant','persistence_mode':'double_json',
        'schedule_clock':'local_completed_updates_independent_of_global_Adam_clock',
        'new_warmup_or_Adam_clock_reset':False,'source_actual_retention_failures':23,'source_actual_selected_step':19750,
        'source_cycle':2,'source_cycle_position':2,'carried_source_cycle_prefix_clips':2,
        'remaining_source_cycle_TR_clips':253796,'declared_next_cycle_tail_TR_draws':10,
        'all207_weights_Adam_sampler_RNG_history_preserved':True,'all_other_math_teacher_inputs_and_replay_unchanged':True,
        'strict_all_six_champion_gates_unchanged':True,'retention_stop_policy':'monitor_only',
        'resource_nonfinite_protections_unchanged':True,'ROOT_review':context['receipt']['ROOT_review'],
        'coverage_rewind_clips':0,'sampler_queues_or_RNG_reset':False,
        'new_cycle_completion_requires_actual_unique_TR_clips':253798,
        'inherited_cumulative_seen_is_not_new_cycle_proof':True,
        'no_plateau_target_or_retention_stop_before_new_cycle_completion':True}
'''+source[end:]
source=source.replace('actual_CLOSED_matched_IO_control','actual_CLOSED_matched_encoder_schedule')
source=source.replace("value.get('persistence_mode') in PERSISTENCE_MODES", "value.get('encoder_lr_schedule') in ENCODER_SCHEDULE_MODES")
source=source.replace('allow_matched_IO_control','allow_matched_encoder_schedule_control')
source=source.replace("review.get('technical_updates')==100 and review.get('technical_end_step')==25209", "review.get('encoder_schedule_updates')==18129 and review.get('encoder_schedule_end_step')==43238\n        and review.get('source_cycle')==2 and review.get('source_cycle_position')==2\n        and review.get('remaining_source_cycle_TR_clips')==253796 and review.get('declared_next_cycle_tail_TR_draws')==10")
start=source.index("    change=review['declared_change']");end=source.index("    closed=read",start)
source=source[:start]+'''    change=review['declared_change']
    require(change.get('parameter')=='encoder_LoRA_schedule' and change.get('choices')==list(ENCODER_SCHEDULE_MODES)
        and change.get('additional_updates')==18129 and change.get('end_step')==43238
        and change.get('source_encoder_lr')==1e-4 and change.get('minimum_encoder_lr')==3e-5
        and change.get('shared_head_lr')==1e-5 and change.get('legacy_schedule')=='constant'
        and change.get('warmup_reset') is False and change.get('persistence_mode')=='double_json'
        and change.get('all_other_recipe_fields_unchanged') is True and change.get('no_coverage_rewind') is True
        and change.get('strict_all_six_winner_gates_unchanged') is True and change.get('retention_stop_policy')=='monitor_only',
        'Only encoder LR schedule and explicit next-shuffle-cycle execution bound may change')
'''+source[end:]
source=source.replace("'io_review':review", "'encoder_schedule_review':review")
source=source.replace("'source_recipe':{**recipe,'steps':25209,'persistence_mode':receipt['persistence_mode']}", "'source_recipe':{**recipe,'steps':43238,'persistence_mode':'double_json',\n            'encoder_lr_schedule':receipt['encoder_lr_schedule'],'encoder_schedule_updates':18129,'encoder_min_lr':3e-5}")
source=source.replace("'remaining_updates_from_chosen_frontier':100", "'remaining_updates_from_chosen_frontier':18129")
source=source.replace("'source_lineage_end_step':25109,'technical_end_step':25209,'technical_updates':100,'technical_extension_updates':100", "'source_lineage_end_step':25109,'encoder_schedule_end_step':43238,'encoder_schedule_updates':18129,\n        'execution_extension_updates':18129,'source_cycle':2,'source_cycle_position':2,\n        'remaining_source_cycle_TR_clips':253796,'declared_next_cycle_tail_TR_draws':10")
source=source.replace("'configured_end_step':25209", "'configured_end_step':43238")
source=source.replace("require(type(args.persistence_benchmark_updates) is int and args.persistence_benchmark_updates==100,\n        'Only declared100-update technical control admitted')", "require(type(args.encoder_schedule_updates) is int and args.encoder_schedule_updates==18129\n        and args.encoder_min_lr==3e-5 and args.persistence_mode=='double_json',\n        'Exact18129-update encoder-only schedule and unchanged persistence required')")
source=source.replace('IO_control_and_original_source_caches_cleared','encoder_schedule_and_original_source_caches_cleared')
helper=base/'tsc_uniform_encoder_schedule_transition_v1.py';assert not helper.exists();ast.parse(source);helper.write_text(source,encoding='utf-8')

creator=(base/'record_TSC_uniform_io_source_v1.py').read_text(encoding='utf-8')
creator=creator.replace('tsc_uniform_io_transition_v1','tsc_uniform_encoder_schedule_transition_v1')
creator=creator.replace('PERSISTENCE_MODES','ENCODER_SCHEDULE_MODES')
creator=creator.replace('allow_matched_IO_control','allow_matched_encoder_schedule_control')
creator=creator.replace('actual_CLOSED_matched_IO_control','actual_CLOSED_matched_encoder_schedule')
creator=creator.replace("'persistence_mode':args.persistence_mode", "'encoder_lr_schedule':args.encoder_lr_schedule")
creator=creator.replace("'technical_end_step':25209", "'encoder_schedule_end_step':43238")
creator=creator.replace("--persistence-mode", "--encoder-lr-schedule")
creator=creator.replace('matched technical IO','matched encoder schedule')
file=base/'record_TSC_uniform_encoder_schedule_source_v1.py';assert not file.exists();ast.parse(creator);file.write_text(creator,encoding='utf-8')
