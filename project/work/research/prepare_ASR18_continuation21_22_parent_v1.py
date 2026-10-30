"""Derive new finite continuation driver without editing admitted parent11."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'work/research'


def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)


text=(R/'run_ASR18_full_DATA_multiwindow_v11.py').read_text(encoding='utf-8')
for old,new in (('asr18_full_DATA_native_window_v5','asr18_full_DATA_native_window_v6'),
    ('asr18_deferred_GPU_recovery_v1','asr18_deferred_GPU_recovery_v2'),
    ('asr18_restore_only_runtime_v1','asr18_restore_only_runtime_v2'),
    ('bpe_final_DATA_schedule_context_v1','bpe_final_DATA_schedule_context_v2')):
    text=replace(text,old,new)
text=replace(text,"and config['maximum_total_native_updates'] == 40 and config['execute_window_limit'] == 1 and type(config['execute_window_limit']) is int",
    "and config['kind']=='ASR18_checkpoint20_two_one_update_windows_diagnostic_config_v1'\n            and config['maximum_total_native_updates']==22 and config['execute_window_limit']==2\n            and type(config['execute_window_limit']) is int and config['maximum_updates_per_window']==1")
text=replace(text,"review['scope'] == 'finite_full_DATA_two_window_diagnostic_v1'",
    "review['scope'] == 'finite_full_DATA_checkpoint20_continuation_two_one_update_windows_v1'")
text=replace(text,"    from bpe_final_DATA_window_planner_v4 import prepare_first_plan\n",'')
text=replace(text,"        first = prepare_first_plan(**config['planning_arguments'], maximum_updates=config['maximum_updates_per_window'], guard=parent.check)",
    "        from asr18_verified_pipeline_ancestor_v1 import verify_ancestor\n        ancestor=verify_ancestor(config['continuation_ancestor'],guard=parent.check)\n        def verified_ancestor(boundary,pointer):\n            require(boundary==config['continuation_ancestor']['actual_TOOL0']\n                    and pointer==config['continuation_ancestor']['checkpoint_pointer'], 'Exact closed ancestor pointer/TOOL0')\n            return digest(verify_ancestor(config['continuation_ancestor'],guard=parent.check))==digest(ancestor)\n        first=prepare_following_plan(config['planning_arguments'],\n            pointer_binding=config['continuation_ancestor']['checkpoint_pointer'],\n            native_boundary_binding=config['continuation_ancestor']['actual_TOOL0'],verify_native_boundary=verified_ancestor,\n            maximum_updates=1,guard=parent.check)")
text=replace(text,"and session['initial_optimizer_updates'] == 0, 'Session must bind actual original-source schedule')",
    "and session['initial_optimizer_updates']==20 and session['ancestor']==config['continuation_ancestor'],\n                'New diagnostic window0 retains exact actual20 scientific state')")
old="""        from asr18_reprepared_array_equivalence_v1 import validate_retirement, verify_repreparation
        replay_check = config['first_window_repreparation_comparison']
        require(replay_check['required_before_any_GPU_launch'] is True,
                'Changed finalizer trial requires deterministic full first-window comparison')
        validate_retirement(replay_check['prior_owned_retirement_actual0'],
            prior_failure=replay_check['prior_failed_trial'], guard=parent.check)
        plan = first['plan']; boundary_pin = None; attestation_pin = None; resume_pointer = None
"""
new="""        require(config['initial_inherited_global_updates']==20 and config['planned_actual_new_updates']==2,
                'Only two NEW global21/22 updates; closed20 is never rerun or reset')
        plan = first['plan']; boundary_pin = None; attestation_pin = None
        resume_pointer=config['continuation_ancestor']['checkpoint_pointer']
"""
text=replace(text,old,new)
old="""            if index == 0:
                verify_repreparation(prior_failure=replay_check['prior_failed_trial'],
                    prior_retirement=replay_check['prior_owned_retirement_actual0'],
                    new_CPU_actual=CPU_actual,new_owner=CPU['owner'],new_manifest=CPU['manifest'],
                    coordinator_plan=plan_pin,output=folder/'first_window_actual_array_repreparation_equivalence.json',
                    guard=parent.check)
"""
text=replace(text,old,"""            require(len(plan['updates'])==1 and plan['initial_committed_updates']==20+index,
                    'One genuine NEW sequential source update per diagnostic window')
""")
text=replace(text,"child_config = {**config, 'restore_parent_pointer': resume_pointer, 'first_feature_update': manifest['updates'][0]['update']}",
    "child_config = {**config, 'restore_parent_pointer': resume_pointer, 'restore_parent_boundary':boundary_pin,\n                    'first_feature_update': manifest['updates'][0]['update']}")
text=replace(text,"'kind':'ASR18_deferred_all_checkpoint_GPU_recovery_requests_v1'",
    "'kind':'ASR18_continuation21_22_all_new_checkpoint_GPU_recovery_requests_v2'")
text=replace(text,"'expected_global_new_updates':sum(value['actual_new_updates'] for value in windows),",
    "'initial_global_updates':20,'expected_new_updates':2,'expected_global_new_updates':22,\n            'ancestor':config['continuation_ancestor'],")
text=replace(text,"result = {'passed': True, 'windows': windows, 'actual_process_completions': actuals, 'resident_selections': selected,",
    "result = {'kind':'ASR18_actual_checkpoint20_continuation21_22_multiwindow_diagnostic_v1',\n            'passed':True,'initial_inherited_global_updates':20,'actual_final_global_updates':22,\n            'ancestor':config['continuation_ancestor'],'windows':windows,'actual_process_completions':actuals,'resident_selections':selected,")
text=replace(text,"'first_window_byte_exact_repreparation_proof':record(folder/'first_window_actual_array_repreparation_equivalence.json'),",
    "'old_firstwindow20_actual0_preserved':config['continuation_ancestor']['actual_TOOL0'],\n            'new_feature_windows_prepared_from_original_sources_without_adoption':True,\n            'one_update_windows_are_diagnostic_not_production_cadence':True,")
path=R/'run_ASR18_full_DATA_multiwindow_v12.py'
assert not path.exists(),path
ast.parse(text)
path.write_text(text,encoding='utf-8',newline='\n')
