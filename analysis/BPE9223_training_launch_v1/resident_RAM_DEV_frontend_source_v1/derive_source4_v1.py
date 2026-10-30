import ast,copy,difflib,json,sys
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish'
N=Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');A=N/'resident_RAM_DEV_frontend_source_v1'
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
old=(R/'run_ASR18_BPE9223_resident_RAM_training_v3.py').read_text();new=old
def sub(before,after):
    global new
    require(new.count(before)==1,'One exact scoped replacement: '+before[:90]);new=new.replace(before,after,1)
checker='''def checked_DEV_frontend_resume(pin,*,base,folder,pointer_pin):
    from fastmss_owned_process_closure_v2 import dead
    require(pin==base['prior_resident_RAM_DEV_frontend_failure'] and record(pin['path'])==pin,
        'Exact reviewed failed periodic frontend owner; no normal0/resource-stop relabel')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_resident_RAM_periodic_DEV_frontend_failure_saved4000_v1'
        and value['actual_exec_tool_exit_code']==1 and value['normal_native0_claimed'] is False
        and value['intentional_stop_claimed'] is False and value['branch']==str(folder)
        and value['checkpoint_pointer']==pointer_pin
        and value['saved_optimizer_updates']==value['actual_optimizer_updates_before_failure']==4000
        and value['unsaved_tail_updates']==[] and value['periodic4000_still_required_before4001'] is True
        and value['actual_WER_report_produced'] is False and all(dead(w)for w in value['captured_identities']),
        'Actual failed first-row audio frontend/dead identities and coherent4000')
    for key in ('original_failure','original_monitor_completion','original_CPU_stderr','whole_journal',
                'checkpoint_pointer','checkpoint_persistence','original_native_log','source_configuration'):
        require(record(value[key]['path'])==value[key],'Preserved source3 failure/saved state/wholejournal')
    for pin in value['failed_results'].values():require(record(pin['path'])==pin,'Old partial DEV predictions/results untouched')
    require(read_json(pointer_pin['path'])['journal_prefix']==value['saved_journal_prefix']
        and value['whole_journal']=={'path':value['saved_journal_prefix']['path'],
            'bytes':value['saved_journal_prefix']['offset'],'sha256':value['saved_journal_prefix']['sha256']},
        'No unsaved tail and no new journal lineage/copy at saved4000')
    require(record(S/'asr18_resident_serial350_DEV_v2.py')==base['resident_DEV_helper_source'],
        'Exact new full-processor helper source')
    return value


'''
sub('class Monitor:',checker+'class Monitor:')
sub("abnormal=None if a.abnormal_resume is None else checked_headroom_resume(record(a.abnormal_resume),base=base,folder=folder,pointer_pin=pointer_pin)",
    "abnormal=None if a.abnormal_resume is None else checked_headroom_resume(record(a.abnormal_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    frontend_failure=None if a.DEV_frontend_resume is None else checked_DEV_frontend_resume(record(a.DEV_frontend_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    require(abnormal is None or frontend_failure is None,'One truthful predecessor failure lane')")
sub("and (a.prior_process is not None or abnormal is not None)","and (a.prior_process is not None or abnormal is not None or frontend_failure is not None)")
sub("        if abnormal is None:\n            prior=read_json(a.prior_process)","        if abnormal is None and frontend_failure is None:\n            prior=read_json(a.prior_process)")
sub("        else:require(a.prior_process is None and a.invocation==4,'Resource failure is not a normal0 predecessor')",
    "        elif abnormal is not None:require(a.prior_process is None and a.invocation==4,'Resource failure is not a normal0 predecessor')\n        else:require(a.prior_process is None and a.invocation==5,'Frontend failure is not a normal0 predecessor')")
sub("        def journal_factory(saved):\n            if abnormal is not None:",
    "        def journal_factory(saved):\n            if frontend_failure is not None:\n                require(saved==source_journal and record(saved['path'])==frontend_failure['whole_journal'],'Exact whole saved4000 journal')\n                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved4000_prefix_restore_after_DEV_frontend_failure_v1',\n                    'failure':record(a.DEV_frontend_resume),'original_modified':False,'new_journal_branch_created':False}\n            if abnormal is not None:")
sub("journal=(SavedJournalPrefixView(source_journal,guard)if abnormal is not None", "journal=(SavedJournalPrefixView(source_journal,guard)if abnormal is not None or frontend_failure is not None")
sub("        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)\n",'')
sub("        completed=len(lifecycle['completed_boundaries']);plan_sequence=0;queued=None;last_checkpoint=initial_updates",
    "        completed=len(lifecycle['completed_boundaries']);plan_sequence=0;queued=None;last_checkpoint=initial_updates\n        latest=read_json(folder/'latest_checkpoint.json')\n        require(latest['pointer']==pointer_pin and latest['full_state_verified'] is True,'Current coherent checkpoint before pending DEV')")
sub('from asr18_resident_serial350_DEV_v1 import evaluate','from asr18_resident_serial350_DEV_v2 import evaluate')
sub("        plan,plan_pin=submit_plan(engine.schedule)\n        while optimizer.updates<c['planned_native_updates']:",
    "        if frontend_failure is not None:\n            require(optimizer.updates==4000 and 4000 not in evaluated and queued is None,'Pending failed4000 evaluation before any4001/planning')\n            lease.update(status='production_resident_RAM_serial350_DEV',actual_training_optimizer_running=False);write_json(lease_path,lease)\n            restored_status.update(phase='pending_periodic_DEV_4000_before4001');write_json(folder/'progress/latest.json',restored_status)\n            periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)\n            dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_000004000_attempt_'+str(a.invocation).zfill(6)))\n            record_periodic(dev_pin,score,preservation,False)\n            require(optimizer.updates==4000 and journal.binding()==source_journal,'Evaluation completed before positive Adam/new journal bytes')\n            lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)\n        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)\n        plan,plan_pin=submit_plan(engine.schedule)\n        while optimizer.updates<c['planned_native_updates']:")
sub("p.add_argument('--prior-process');p.add_argument('--abnormal-resume');p.add_argument('--validation-stop',type=int)",
    "p.add_argument('--prior-process');p.add_argument('--abnormal-resume');p.add_argument('--DEV-frontend-resume');p.add_argument('--validation-stop',type=int)")
ast.parse(new);target=R/'run_ASR18_BPE9223_resident_RAM_training_v4.py';require(not target.exists(),'New source4');target.write_text(new,encoding='utf-8')
(A/'driver3_to4.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='driver3',tofile='driver4')),encoding='utf-8')

old=(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v3.py').read_text();new=old
sub('resident_RAM_training_v3.py','resident_RAM_training_v4.py')
sub("and review['CPU_source']==record(S/'fastmss_resident_feature_stream_v1.py')", "and review['CPU_source']==record(S/'fastmss_resident_feature_stream_v1.py')\n        and review['DEV_helper_source']==record(S/'asr18_resident_serial350_DEV_v2.py')")
sub("and a.abnormal_resume is not None and review['abnormal_resume']==record(a.abnormal_resume)",
    "and a.DEV_frontend_resume is not None and review['DEV_frontend_resume']==record(a.DEV_frontend_resume)")
sub("elif invocation==a.start_invocation:command+=['--abnormal-resume',a.abnormal_resume]", "elif invocation==a.start_invocation:command+=['--DEV-frontend-resume',a.DEV_frontend_resume]")
sub("p.add_argument('--prior-process');p.add_argument('--abnormal-resume');main(p.parse_args())", "p.add_argument('--prior-process');p.add_argument('--DEV-frontend-resume');main(p.parse_args())")
ast.parse(new);wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v4.py';require(not wrapper.exists(),'New wrapper4');wrapper.write_text(new,encoding='utf-8')
(A/'wrapper3_to4.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='wrapper3',tofile='wrapper4')),encoding='utf-8')
configuration=read_json(N/'resident_RAM_periodic4000_source_v1/configuration.json')
configuration['prior_resident_RAM_DEV_frontend_failure']=record(N/'resident_RAM_DEV_frontend_failure_actual_v1/actual_failure_closure.json')
configuration['resident_DEV_helper_source']=record(S/'asr18_resident_serial350_DEV_v2.py')
write_json(A/'configuration.json',configuration)
handoff={'kind':'ASR18_minimal_real_DEV_frontend_fix_saved4000_source_handoff_v1',
    'driver':record(target),'wrapper':record(wrapper),'helper':configuration['resident_DEV_helper_source'],'configuration':record(A/'configuration.json'),
    'actual_failure':configuration['prior_resident_RAM_DEV_frontend_failure'],
    'CPU_composition_report':record(N/'resident_RAM_DEV_frontend_composition_actual_01/report.json'),
    'CPU_composition_monitor':record(N/'resident_RAM_DEV_frontend_composition_actual_01/monitor_completion.json'),
    'CPU_composition_actual_exec_exit_code':0,'CPU_composition_actual_exec_chunk':'283cde',
    'CPU_composition_fixture':record(A/'real_CPU_composition_fixture_v1.py'),
    'checkpoint_pointer':read_json(configuration['prior_resident_RAM_DEV_frontend_failure']['path'])['checkpoint_pointer'],
    'launch_invocation':5,'queue_bytes':268435456,'full_native_target_updates':401072,
    'same_active_journal_reused_after_readonly_restore':True,'pending4000_before_any4001_or_preparation':True,
    'failed_DEV_outputs_preserved':True,'config_delta_keys':['prior_resident_RAM_DEV_frontend_failure','resident_DEV_helper_source'],
    'diffs':{k:record(A/k)for k in ('helper1_to2.diff.txt','driver3_to4.diff.txt','wrapper3_to4.diff.txt')}}
write_json(A/'source_handoff.json',handoff);print(json.dumps(handoff))
