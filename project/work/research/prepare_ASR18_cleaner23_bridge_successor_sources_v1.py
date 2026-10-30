"""Emit NEW cleaner caller import/binding successors; no target module import."""
import ast
from pathlib import Path

P=Path(__file__).resolve().parents[2]
S=P/'outputs/parakeet_turkish'
R=P/'work/research'

def emit(source,target,replacements):
    assert not target.exists(), 'Preserve every frozen source'
    text=source.read_text(encoding='utf-8-sig')
    for old,new in replacements:
        assert old in text, (source,old)
        text=text.replace(old,new)
    ast.parse(text,filename=str(target))
    with target.open('x',encoding='utf-8',newline='\r\n') as stream:stream.write(text)
    print(str(target),flush=True)

emit(S/'asr18_targeted_cleaner23_native_v1.py',S/'asr18_targeted_cleaner23_native_v2.py',[
    ('fastmss_phase_aware_feature_window_v4','fastmss_phase_aware_feature_window_v5')])
emit(S/'asr18_targeted_cleaner23_recovery_v1.py',S/'asr18_targeted_cleaner23_recovery_v2.py',[
    ('fastmss_phase_aware_plan_bridge_v5','fastmss_phase_aware_plan_bridge_v6')])
emit(S/'asr18_targeted_cleaner23_restore_runtime_v1.py',S/'asr18_targeted_cleaner23_restore_runtime_v2.py',[
    ('asr18_targeted_cleaner23_recovery_v1','asr18_targeted_cleaner23_recovery_v2')])
emit(R/'run_ASR18_targeted_cleaner23_v1.py',R/'run_ASR18_targeted_cleaner23_v2.py',[
    ('asr18_targeted_cleaner23_native_v1','asr18_targeted_cleaner23_native_v2'),
    ('asr18_targeted_cleaner23_recovery_v1','asr18_targeted_cleaner23_recovery_v2'),
    ('asr18_targeted_cleaner23_restore_runtime_v1','asr18_targeted_cleaner23_restore_runtime_v2'),
    ('fastmss_phase_aware_feature_window_v4','fastmss_phase_aware_feature_window_v5'),
    ('fastmss_phase_aware_plan_bridge_v5','fastmss_phase_aware_plan_bridge_v6'),
    ('fastmss_row_serial_cpu_v6','fastmss_row_serial_cpu_v7'),
    ('FastMSS_row_serial_CPU_preparation_contract_v6','FastMSS_row_serial_CPU_preparation_contract_v7'),
    ('finite_targeted_cleaner23_CPU6_native_and_fresh_recovery_diagnostic_v1','finite_targeted_cleaner23_CPU7_native_and_fresh_recovery_diagnostic_v2'),
    ("first=context.plan(initial,maximum_updates=1)\n", "first=context.plan(initial,maximum_updates=1)\n            first['targeted_cleaner_diagnostic_context']=config['targeted_cleaner_diagnostic_context']\n"),
    ("session=read_bound(config['session'],parent.check);CPU_contract=read_bound(session['CPU_preparation_contract'],parent.check)", "session=read_bound(config['session'],parent.check);CPU_contract=read_bound(session['CPU_preparation_contract'],parent.check)\n        require(session['targeted_cleaner_diagnostic_context']==config['targeted_cleaner_diagnostic_context'],\n                'Session and caller bind the same explicit diagnostic cleaner context')"),
    ("ASR18_actual_targeted_cleaner23_CPU6_native_fresh_recovery_diagnostic_v1", "ASR18_actual_targeted_cleaner23_CPU7_native_fresh_recovery_diagnostic_v2")])
emit(R/'prepare_ASR18_targeted_cleaner23_configuration_v1.py',R/'prepare_ASR18_targeted_cleaner23_configuration_v2.py',[
    ('fastmss_row_serial_cpu_v6','fastmss_row_serial_cpu_v7'),
    ('fastmss_phase_aware_plan_bridge_v5','fastmss_phase_aware_plan_bridge_v6'),
    ('FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v6.json','FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v7.json'),
    ('FastMSS_row_serial_CPU_preparation_contract_v6','FastMSS_row_serial_CPU_preparation_contract_v7'),
    ("from fastmss_phase_aware_plan_bridge_v6 import read_bound,PURPOSE,SESSION_KIND", "from fastmss_phase_aware_plan_bridge_v6 import read_bound,PURPOSE,SESSION_KIND,bridge_plan"),
    ("schedule=context.new_schedule();initial=copy.deepcopy(schedule.state_dict());plan=context.plan(schedule,maximum_updates=1)", "schedule=context.new_schedule();initial=copy.deepcopy(schedule.state_dict())\n        folder.mkdir()\n        context_document={'kind':'ASR18_targeted_cleaner23_bridge_context_v1',\n            'ancestor':template['cleaner_ancestor'],'initial_diagnostic_sampler':initial,\n            **{key:template['planning_arguments'][key] for key in ('snapshot_binding','snapshot_actual0','snapshot_report','expected_original_sources','replay_pool_binding')},\n            'schedule_adapter_source':record(S/'asr18_targeted_cleaner_schedule_context_v1.py')}\n        context_path=folder/'targeted_cleaner_diagnostic_context.json';write_json(context_path,context_document);context_pin=record(context_path)\n        plan=context.plan(schedule,maximum_updates=1);plan['targeted_cleaner_diagnostic_context']=context_pin"),
    ("folder.mkdir();plan_path=folder/'targeted_cleaner23_plan.json'", "plan_path=folder/'targeted_cleaner23_plan.json'"),
    ("'initial_sampler_sha256':plan['initial_sampler_sha256'],'ancestor':template['cleaner_ancestor'],", "'initial_sampler_sha256':plan['initial_sampler_sha256'],'ancestor':template['cleaner_ancestor'],\n        'targeted_cleaner_diagnostic_context':context_pin,"),
    ("sp=folder/'targeted_cleaner23_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)", "sp=folder/'targeted_cleaner23_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)\n    bridged=bridge_plan(record(plan_path),DATA_evidence=template['planning_arguments']['DATA_evidence'],\n        session_binding=record(sp),window_index=0,prior_window_boundary=None,guard=guard)\n    require(bridged['phase']==plan['phase'] and len(bridged['updates'])==1 and bridged['initial_committed_updates']==22\n            and bridged['final_sampler_sha256']==plan['updates'][0]['after_sampler_sha256'],\n            'Actual metadata-only bridge call must accept the same complete diagnostic cleaner transaction before audio')\n    bridge_path=folder/'actual_metadata_bridged_cleaner_plan.json';write_json(bridge_path,bridged)"),
    ("'initial_cleaner_diagnostic_sampler_sha256':plan['initial_sampler_sha256'],'provisional_cleaner_plan':record(plan_path),", "'initial_cleaner_diagnostic_sampler_sha256':plan['initial_sampler_sha256'],'provisional_cleaner_plan':record(plan_path),\n        'targeted_cleaner_diagnostic_context':context_pin,'actual_metadata_bridged_cleaner_plan':record(bridge_path),"),
    ("'provisional_cleaner_plan':record(plan_path),'source_manifest':source_manifest,'source_review':source_review,", "'provisional_cleaner_plan':record(plan_path),'source_manifest':source_manifest,'source_review':source_review,\n        'targeted_cleaner_diagnostic_context':context_pin,'actual_metadata_bridged_cleaner_plan':record(bridge_path),\n        'actual_metadata_bridge_plan_call_passed':True,"),
    ('ASR18_targeted_cleaner23_actual_metadata_configuration_v1','ASR18_targeted_cleaner23_actual_metadata_configuration_v2')])
