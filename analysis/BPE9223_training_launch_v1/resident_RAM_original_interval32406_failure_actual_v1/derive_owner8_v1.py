import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish';N=Path(__file__).parent
p=R/'run_ASR18_BPE9223_resident_RAM_training_v7.py';s=p.read_text();old=s
check='''def checked_original_interval_resume(pin,*,base,folder,pointer_pin):
    """Exact actual CPU interval failure; never a normal0 or intentional stop."""
    import psutil
    from fastmss_owned_process_closure_v2 import dead
    require(pin==base['prior_resident_RAM_original_interval_failure'] and record(pin['path'])==pin,
        'Exact reviewed original-interval failure')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_resident_RAM_original_interval_failure_saved32400_tail32405_v1'
        and value['actual_owned_session']==53376 and value['actual_exec_tool_exit_code']==1
        and value['actual_wrapper_command_exit_code']==1
        and value['actual_native_OS_exit_code'] is None and value['actual_CPU_OS_exit_code'] is None
        and value['actual_native_nonzero_wait_observed_by_wrapper'] is True
        and value['normal_native0_claimed'] is False and value['intentional_stop_claimed'] is False
        and value['branch']==str(folder) and value['checkpoint_pointer']==pointer_pin
        and value['saved_optimizer_updates']==32400 and value['actual_optimizer_updates_before_failure']==32405
        and value['preserved_tail_positive_updates']==list(range(32401,32406))
        and all(dead(w)for w in value['captured_identity_pairs'])
        and all(not psutil.pid_exists(w['pid']) for w in value['known_old_PIDs_without_recorded_create_time']),
        'Actual failed owner dead, coherent32400 and five later positive attempts remain historical')
    for key in ('original_failure','original_monitor_completion','original_CPU_stderr','original_CPU_ready',
                'original_native_log','original_current_child','source_configuration','effective_configuration',
                'worker_source','ROOT_execution_review','checkpoint_pointer','checkpoint_persistence',
                'whole_journal_including_preserved_tail','preserved_unsaved_tail','failed_preparation_plan','failed_update_metadata'):
        require(record(value[key]['path'])==value[key],'Preserved actual failed source/state/tail evidence')
    failure=read_json(value['original_failure']['path']);monitor=read_json(value['original_monitor_completion']['path'])
    require('Original interval beyond source unresolved' in failure['error']
        and failure['actual_optimizer_updates']==32405 and failure['last_checkpoint']==32400
        and monitor['joined'] is True and monitor['error'] is None
        and read_json(pointer_pin['path'])['journal_prefix']==value['saved_journal_prefix'],
        'Actual nonresource CPU EOF failure and unchanged saved journal authority')
    return value


'''
assert s.count('def private_successor_request(')==1;s=s.replace('def private_successor_request(',check+'def private_successor_request(',1)
def replace(a,b):
 global s
 assert s.count(a)==1,(s.count(a),a);s=s.replace(a,b)
replace("    require(sum(x is not None for x in (abnormal,frontend_failure,transition))<=1,'One truthful predecessor lane')", "    interval_failure=None if a.original_interval_resume is None else checked_original_interval_resume(\n        record(a.original_interval_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    require(sum(x is not None for x in (abnormal,frontend_failure,transition,interval_failure))<=1,'One truthful predecessor lane')")
replace("    preserved_tail=abnormal if abnormal is not None else(transition if transition is not None and transition['unsaved_tail_bytes'] else None)","    preserved_tail=interval_failure if interval_failure is not None else(abnormal if abnormal is not None else(transition if transition is not None and transition['unsaved_tail_bytes'] else None))")
replace("(a.prior_process is not None or abnormal is not None or frontend_failure is not None or transition is not None)","(a.prior_process is not None or abnormal is not None or frontend_failure is not None or transition is not None or interval_failure is not None)")
replace("        if abnormal is None and frontend_failure is None and transition is None:","        if abnormal is None and frontend_failure is None and transition is None and interval_failure is None:")
replace("        else:require(a.prior_process is None and a.invocation>5,'Intentional TOOL1 source transition is never normal0')","        elif interval_failure is not None:require(a.prior_process is None and a.invocation==7,'Actual original-interval TOOL1 is not normal0/intentional stop')\n        else:require(a.prior_process is None and a.invocation>5,'Intentional TOOL1 source transition is never normal0')")
replace("    if abnormal is None and transition is None:","    if abnormal is None and transition is None and interval_failure is None:")
replace("require(source_journal==(abnormal if abnormal is not None else transition)['saved_journal_prefix']", "require(source_journal==(interval_failure if interval_failure is not None else(abnormal if abnormal is not None else transition))['saved_journal_prefix']")
replace("        def journal_factory(saved):\n", "        def journal_factory(saved):\n            if interval_failure is not None:\n                require(saved==source_journal,'Exact actual32400 saved prefix, preserved five-tail never replayed')\n                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved32400_prefix_after_original_interval_failure_v1',\n                    'failure':record(a.original_interval_resume),'original_tail_modified':False,'no_tail_replay':True}\n")
replace("if abnormal is not None or frontend_failure is not None or transition is not None\n", "if abnormal is not None or frontend_failure is not None or transition is not None or interval_failure is not None\n")
replace("record(a.abnormal_resume if abnormal is not None else a.source_transition_resume)","record(a.original_interval_resume if interval_failure is not None else(a.abnormal_resume if abnormal is not None else a.source_transition_resume))")
replace('from fastmss_resident_feature_stream_v1 import ResidentFeatureProducer','from fastmss_resident_feature_stream_v2 import ResidentFeatureProducer')
replace("p.add_argument('--source-transition-resume');p.add_argument('--prior-process')", "p.add_argument('--original-interval-resume');p.add_argument('--source-transition-resume');p.add_argument('--prior-process')")
new=R/'run_ASR18_BPE9223_resident_RAM_training_v8.py';assert not new.exists();new.write_text(s,encoding='utf-8');ast.parse(s)
(N/'driver7_to8.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(p),tofile=str(new))),encoding='utf-8')
p=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v7.py';s=p.read_text();old=s
replace('run_ASR18_BPE9223_resident_RAM_training_v7.py','run_ASR18_BPE9223_resident_RAM_training_v8.py')
replace('fastmss_resident_feature_stream_v1.py','fastmss_resident_feature_stream_v2.py')
replace("        require(a.source_transition_resume is None and a.DEV_frontend_resume is None,", "        require(a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None,")
replace("    elif a.source_transition_resume is not None:","    elif a.original_interval_resume is not None:\n        require(a.source_transition_resume is None and a.DEV_frontend_resume is None\n            and review['original_interval_resume']==record(a.original_interval_resume)\n            and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay'],\n            'Exact reviewed source-bound physical geometry correction and actual abnormal32400 predecessor')\n    elif a.source_transition_resume is not None:")
replace("    if a.source_transition_resume is not None:\n        from", "    if a.original_interval_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v8 import checked_original_interval_resume\n        checked_original_interval_resume(record(a.original_interval_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    if a.source_transition_resume is not None:\n        from")
replace("            if a.source_transition_resume is not None:command+=['--source-transition-resume',a.source_transition_resume]", "            if a.original_interval_resume is not None:command+=['--original-interval-resume',a.original_interval_resume]\n            elif a.source_transition_resume is not None:command+=['--source-transition-resume',a.source_transition_resume]")
replace("p.add_argument('--prior-process');p.add_argument('--source-transition-resume')", "p.add_argument('--prior-process');p.add_argument('--original-interval-resume');p.add_argument('--source-transition-resume')")
new=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v8.py';assert not new.exists();new.write_text(s,encoding='utf-8');ast.parse(s)
(N/'wrapper7_to8.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(p),tofile=str(new))),encoding='utf-8')
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
h={key:pin(path) for key,path in (('driver',R/'run_ASR18_BPE9223_resident_RAM_training_v8.py'),('wrapper',new),('driver_delta',N/'driver7_to8.diff.txt'),('wrapper_delta',N/'wrapper7_to8.diff.txt'))}
(N/'driver8_incremental_source_handoff.json').write_text(json.dumps(h,indent=2)+'\n',encoding='utf-8');print(json.dumps(h))
