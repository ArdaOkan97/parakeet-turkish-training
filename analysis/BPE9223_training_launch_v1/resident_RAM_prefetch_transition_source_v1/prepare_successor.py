"""Generate only inactive source6 successors from the frozen reviewed source5."""
import ast, copy, difflib, hashlib, json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20'); R=P/'work/research'
N=Path(__file__).parent
def pin(p):
 p=Path(p); b=p.read_bytes(); return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def replace(s,old,new):
 assert s.count(old)==1,old[:120]
 return s.replace(old,new)
old=R/'run_ASR18_BPE9223_resident_RAM_training_v5.py'; new=R/'run_ASR18_BPE9223_resident_RAM_training_v6.py'
assert not new.exists(); s=old.read_text(encoding='utf-8')
s=replace(s,'def adopt_private_successor(prefetch,ticket,schedule,submit_plan,*,wait):','def adopt_private_successor(prefetch,ticket,schedule,submit_plan,*,wait,timings=None):')
s=replace(s,'    if not wait and not prefetch.ready():return None\n    prepared=prefetch.take(ticket,wait=wait)', '''    if not wait and not prefetch.ready():
        if timings is not None:timings['not_ready_polls']+=1
        return None
    began=time.monotonic();prepared=prefetch.take(ticket,wait=wait);taken=time.monotonic()
    if timings is not None:
        timings.update(worker_preparation_seconds=prepared['preparation_seconds'],worker_metadata_bytes=prepared['metadata_bytes'],
            main_take_wait_seconds=taken-began,end_window_wait_count=int(wait))''')
s=replace(s,"            'Prepared successor begins at the exact private future state')\n        return queued", "            'Prepared successor begins at the exact private future state')\n        if timings is not None:timings['main_adoption_and_submission_seconds']=time.monotonic()-taken\n        return queued")
s=replace(s,"    require(abnormal is None or frontend_failure is None,'One truthful predecessor failure lane')", """    from asr18_resident_source_transition_v1 import checked_source_transition_resume
    transition=None if a.source_transition_resume is None else checked_source_transition_resume(
        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    require(sum(x is not None for x in (abnormal,frontend_failure,transition))<=1,'One truthful predecessor lane')
    preserved_tail=abnormal if abnormal is not None else(transition if transition is not None and transition['unsaved_tail_bytes'] else None)""")
s=replace(s,'and (a.prior_process is not None or abnormal is not None or frontend_failure is not None),','and (a.prior_process is not None or abnormal is not None or frontend_failure is not None or transition is not None),')
s=replace(s,'        if abnormal is None and frontend_failure is None:\n            prior=', '        if abnormal is None and frontend_failure is None and transition is None:\n            prior=')
s=replace(s,"        else:require(a.prior_process is None and a.invocation==5,'Frontend failure is not a normal0 predecessor')", "        elif frontend_failure is not None:require(a.prior_process is None and a.invocation==5,'Frontend failure is not a normal0 predecessor')\n        else:require(a.prior_process is None and a.invocation>5,'Intentional TOOL1 source transition is never normal0')")
s=replace(s,"    if abnormal is None:\n        require(record(source_journal['path'])", "    if abnormal is None and transition is None:\n        require(record(source_journal['path'])")
s=replace(s,"        require(source_journal==abnormal['saved_journal_prefix'],'Exact actual3060 saved prefix')", "        require(source_journal==(abnormal if abnormal is not None else transition)['saved_journal_prefix'],'Exact reviewed saved prefix')")
s=replace(s,'        def journal_factory(saved):\n            if frontend_failure is not None:', """        def journal_factory(saved):
            if transition is not None:
                require(saved==source_journal,'Exact intentionally closed owner saved prefix')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_source_transition_saved_prefix_restore_v1',
                    'actual_transition':record(a.source_transition_resume),'no_tail_replay':True,'original_modified':False}
            if frontend_failure is not None:""")
s=replace(s,'if abnormal is not None or frontend_failure is not None\n                else source_namespace.writer', 'if abnormal is not None or frontend_failure is not None or transition is not None\n                else source_namespace.writer')
s=replace(s,"        if abnormal is not None:\n            destination=source_namespace.register", "        if preserved_tail is not None:\n            destination=source_namespace.register")
s=replace(s,"require(record(source_journal['path'])==abnormal['whole_journal_including_preserved_tail'],'Old tail byteexact after prefix copy')", "require(record(source_journal['path'])==preserved_tail['whole_journal_including_preserved_tail'],'Old tail byteexact after prefix copy')")
s=replace(s,"'kind':'ASR18_actual_saved3060_recovery_journal_branch_v1',\n                'failure':record(a.abnormal_resume),'saved_prefix':source_journal,'preserved_original':abnormal['whole_journal_including_preserved_tail'],\n                'new_active_journal':journal.binding(),'historical_unsaved_tail_updates':list(range(3061,3070)),", "'kind':'ASR18_actual_saved_prefix_recovery_journal_branch_v1',\n                'predecessor':record(a.abnormal_resume if abnormal is not None else a.source_transition_resume),\n                'saved_prefix':source_journal,'preserved_original':preserved_tail['whole_journal_including_preserved_tail'],\n                'new_active_journal':journal.binding(),'historical_unsaved_tail_updates':preserved_tail['preserved_tail_positive_updates'],")
s=replace(s,"            validate_window_plan(engine.schedule.state_dict(),action,plan)\n            private=", """            validate_window_plan(engine.schedule.state_dict(),action,plan)
            window_started=time.monotonic();window_native_before=native_seconds;window_audio_before=audio_seconds
            metadata_timing={'not_ready_polls':0,'end_window_wait_count':0,'serial_fallback_count':0}
            private=""")
s=replace(s,"                else:queued=submit_plan(private) # Original serial cleaner path, not a broadened source contract.", """                else:
                    planning_started=time.monotonic();queued=submit_plan(private)
                    metadata_timing.update(serial_fallback_count=1,serial_planning_and_submission_seconds=time.monotonic()-planning_started)""")
s=replace(s,'queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=False)', 'queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=False,timings=metadata_timing)')
s=replace(s,'queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=True)', 'queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=True,timings=metadata_timing)')
s=replace(s,'            if need_checkpoint:\n                E_reserve', '            checkpoint_started=time.monotonic()\n            if need_checkpoint:\n                E_reserve')
s=replace(s,"            action=next_boundary_request(engine.schedule.state_dict(),completed)\n            if action['kind']=='checkpoint_then_serial350DEV':", """            metadata_timing.update(kind='ASR18_actual_private_metadata_window_timing_v1',
                initial_updates=plan['initial_committed_updates'],committed_updates=optimizer.updates,
                native_seconds=native_seconds-window_native_before,source_audio_seconds=audio_seconds-window_audio_before,
                checkpoint_included=need_checkpoint,checkpoint_seconds=time.monotonic()-checkpoint_started if need_checkpoint else 0.,
                window_pipeline_seconds=time.monotonic()-window_started)
            with(invocation/'private_metadata_window_timings.jsonl').open('a',encoding='utf-8')as timing_log:
                timing_log.write(json.dumps(metadata_timing,separators=(',',':'))+'\\n')
            action=next_boundary_request(engine.schedule.state_dict(),completed)
            if action['kind']=='checkpoint_then_serial350DEV':""")
s=replace(s,"p.add_argument('--prior-process');p.add_argument('--abnormal-resume');", "p.add_argument('--source-transition-resume');p.add_argument('--prior-process');p.add_argument('--abnormal-resume');")
ast.parse(s);new.write_text(s,encoding='utf-8',newline='\n')
N.joinpath('driver5_to6.diff.txt').write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),s.splitlines(True),fromfile=old.name,tofile=new.name)),encoding='utf-8')
oldw=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v5.py';neww=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v6.py';assert not neww.exists()
w=oldw.read_text(encoding='utf-8').replace('run_ASR18_BPE9223_resident_RAM_training_v5.py','run_ASR18_BPE9223_resident_RAM_training_v6.py')
w=replace(w,"    if a.prior_process is not None:\n        require(a.DEV_frontend_resume is None", "    if a.prior_process is not None:\n        require(a.source_transition_resume is None and a.DEV_frontend_resume is None")
w=replace(w,"    else:\n        require(a.DEV_frontend_resume is not None and review['DEV_frontend_resume']==record(a.DEV_frontend_resume),", """    elif a.source_transition_resume is not None:
        require(a.DEV_frontend_resume is None and review['source_transition_resume']==record(a.source_transition_resume)
            and review['source_transition_validator']==record(S/'asr18_resident_source_transition_v1.py'),
            'Exact reviewed actual intentional TOOL1 transition, no normal0 fiction')
    else:
        require(a.DEV_frontend_resume is not None and review['DEV_frontend_resume']==record(a.DEV_frontend_resume),""")
w=replace(w,"    require(control.is_relative_to(Path(r'E:\\gpu_training_scratchpad')) and not control.exists(),'Fresh E loop records');control.mkdir()", """    if a.source_transition_resume is not None:
        from asr18_resident_source_transition_v1 import checked_source_transition_resume
        checked_source_transition_resume(record(a.source_transition_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))
    require(control.is_relative_to(Path(r'E:\gpu_training_scratchpad')) and not control.exists(),'Fresh E loop records');control.mkdir()""")
w=replace(w,"        elif invocation==a.start_invocation:command+=['--DEV-frontend-resume',a.DEV_frontend_resume]", """        elif invocation==a.start_invocation:
            if a.source_transition_resume is not None:command+=['--source-transition-resume',a.source_transition_resume]
            else:command+=['--DEV-frontend-resume',a.DEV_frontend_resume]""")
w=replace(w,"p.add_argument('--prior-process');p.add_argument('--DEV-frontend-resume');", "p.add_argument('--prior-process');p.add_argument('--source-transition-resume');p.add_argument('--DEV-frontend-resume');")
ast.parse(w);neww.write_text(w,encoding='utf-8',newline='\n')
N.joinpath('wrapper5_to6.diff.txt').write_text(''.join(difflib.unified_diff(oldw.read_text(encoding='utf-8').splitlines(True),w.splitlines(True),fromfile=oldw.name,tofile=neww.name)),encoding='utf-8')
print(json.dumps({'driver':pin(new),'wrapper':pin(neww)}))
