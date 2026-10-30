"""Prepare inactive successors only; never imports a training owner or starts work."""
import ast, difflib, hashlib, json
from pathlib import Path

P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish'
N=Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');O=Path(__file__).resolve().parent

def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def replace(s,old,new):
    assert s.count(old)==1, repr(old[:120])
    return s.replace(old,new)

old_driver=R/'run_ASR18_BPE9223_resident_RAM_training_v4.py'
new_driver=R/'run_ASR18_BPE9223_resident_RAM_training_v5.py'
old_wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v4.py'
new_wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v5.py'
assert not new_driver.exists() and not new_wrapper.exists()
s=old_driver.read_text(encoding='utf-8')
s=replace(s,'class Monitor:', '''def private_successor_request(state,*,completed_boundaries,validation_stop,budget_stop):
    """Original MainThread pass bounds, plus the unchanged periodic stop."""
    if validation_stop or budget_stop:return None
    from asr18_BPE9223_lifecycle_v3 import next_boundary_request
    request=next_boundary_request(state,completed_boundaries)
    if request['kind']!='bounded_training_window':return None
    remaining=4000-request['initial_optimizer_updates']%4000
    return {'maximum_updates':min(request['maximum_updates'],remaining),
        'before_sampler_sha256':digest(state),'phase':request['phase']}


def adopt_private_successor(prefetch,ticket,schedule,submit_plan,*,wait):
    """Only ordered full-row ingress changes; original planning stays MainThread."""
    if not wait and not prefetch.ready():return None
    prepared=prefetch.take(ticket,wait=wait)
    try:
        with prefetch.planning_rows(prepared,schedule):
            queued=submit_plan(schedule)
        require(queued[0]['initial_sampler_sha256']==digest(schedule.state_dict()),
            'Prepared successor begins at the exact private future state')
        return queued
    finally:del prepared


class Monitor:''')
s=replace(s,"    initial_updates=pointer['committed_updates']\n", """    initial_updates=pointer['committed_updates']
    prefetch_policy=base['private_scalar_prefetch_policy']
    require(prefetch_policy=={'kind':'ASR18_private_scalar_row_prefetch_policy_v1',
        'source':record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
        'actual_CPU_control_review':record(ROOT/'analysis/BPE9223_training_launch_v1/ROOT_private_scalar_prefetch_actual_CPU_control_review_v1.json'),
        'additional_host_reservation_bytes':268435456,'maximum_TR_rows':280,'maximum_canonical_row_bytes':8388608,
        'SQLite_cache_bytes_each':4194304,'supported_phase':'weak_label_continued_training',
        'original_MainThread_planning_and_commits_unchanged':True,'idle_during_DEV':True},
        'Exact inactive-reviewed scalar prefetch source/resource policy; production still needs execution review')
""")
s=replace(s,"    monitor=Monitor(invocation);stream=None;engine=None;torch=None;began=time.monotonic();stage_stats={};scopes=ExitStack()\n", """    monitor=Monitor(invocation);stream=None;engine=None;torch=None;began=time.monotonic();stage_stats={};scopes=ExitStack()
    prefetch=None;prefetch_close_attempted=False
    def close_private_metadata():
        nonlocal prefetch_close_attempted
        if prefetch is None or prefetch_close_attempted:return
        prefetch_close_attempted=True
        try:
            proof=prefetch.close(discard_unconsumed=True)
            write_json(invocation/'private_metadata_worker_completion.json',dict(proof,source=prefetch_policy['source']))
        except BaseException as error:
            write_json(invocation/'private_metadata_worker_close_failure.json',{'error':repr(error),
                'worker_thread_alive':prefetch.thread.is_alive(),'worker_status_closed':prefetch.worker_status['closed'],
                'normal_worker_closure_claimed':False})
            raise
    def release_current_lease(committed):
        current=read_json(lease_path)
        if current.get('actual_owned_worker_pid')==os.getpid():
            current.update(running=False,actual_GPU_running=False,actual_training_optimizer_running=False,actual_owned_worker_pid=None,
                status='production_resident_RAM_process_closed',committed_updates=committed);write_json(lease_path,current)
""")
s=replace(s,"""        def admit(name,increment):
            guard();s=monitor._observe();require(s['system_memory']['commit_available_bytes']-increment>=2000000000,
                'Actual allocation increment plus real system headroom')
            return {'passed':True,'increment':increment,'sample':s}
""", """        def admit(name,increment):
            guard();s=monitor._observe()
            reserve=prefetch_policy['additional_host_reservation_bytes']
            extra=0 if name=='private_scalar_prefetch_additional_host_state' else reserve
            require(s['system_memory']['commit_available_bytes']-increment-extra>=2000000000,
                'Actual allocation increment plus full extra prefetch reservation and unchanged commit headroom')
            require(s['system_memory']['physical_available_bytes']-reserve>=1500000000,
                'Full extra256MiB host reservation plus unchanged real physical headroom; no freed-memory credit')
            return {'passed':True,'increment':increment,'additional_prefetch_reservation_bytes':extra,'sample':s}
""")
s=replace(s,"""        plan,plan_pin=submit_plan(engine.schedule)
        while optimizer.updates<c['planned_native_updates']:
""", """        plan,plan_pin=submit_plan(engine.schedule)
        from asr18_BPE9223_private_scalar_prefetch_v1 import PrivateScalarPrefetch
        if engine.schedule.phase=='weak_label_continued_training':
            prefetch=PrivateScalarPrefetch(engine.context,guard=guard,admit=admit)
            write_json(invocation/'private_metadata_worker_admission.json',prefetch.admission)
        while optimizer.updates<c['planned_native_updates']:
""")
s=replace(s,"""            if not(validation_stop or budget_stop)and next_boundary_request(private.state_dict(),completed)['kind']=='bounded_training_window':queued=submit_plan(private)
            else:queued=None
""", """            successor=private_successor_request(private.state_dict(),completed_boundaries=completed,
                validation_stop=validation_stop,budget_stop=budget_stop)
            queued=None;future_ticket=None
            if successor is not None:
                if prefetch is not None and successor['phase']=='weak_label_continued_training':
                    # Recheck fresh combined headroom before each new bounded private request.
                    admit('private_scalar_prefetch_additional_host_state',prefetch_policy['additional_host_reservation_bytes'])
                    future_ticket=prefetch.submit(private.state_dict(),maximum_updates=successor['maximum_updates'])
                else:queued=submit_plan(private) # Original serial cleaner path, not a broadened source contract.
""")
s=replace(s,"""                prepared_archive.append({'binding':binding,'metadata':metadata,'native':native});stream.release(packet['token']);del packet,arrays
""", """                prepared_archive.append({'binding':binding,'metadata':metadata,'native':native});stream.release(packet['token']);del packet,arrays
                if future_ticket is not None:
                    queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=False)
                    if queued is not None:future_ticket=None
""")
s=replace(s,"""            periodic_due=optimizer.updates>=4000 and optimizer.updates%4000==0 and optimizer.updates not in evaluated
""", """            if future_ticket is not None:
                queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=True)
                future_ticket=None
            require(prefetch is None or(prefetch.outstanding is None and prefetch.held_result is None),
                'All ordered metadata adopted and submitted; no active/pending metadata during checkpoint or DEV')
            del private
            periodic_due=optimizer.updates>=4000 and optimizer.updates%4000==0 and optimizer.updates not in evaluated
""")
s=replace(s,"""        require(last_checkpoint==optimizer.updates,'Normal close saves every live update')
""", """        require(last_checkpoint==optimizer.updates,'Normal close saves every live update')
        close_private_metadata() # Genuine joined closure before normal0/engine context closure.
""")
s=replace(s,"""    finally:
        if stream is not None:stream.close()
        scopes.close()
        if engine is not None:engine.close()
        monitor.close()
        lease=read_json(lease_path)
        if lease.get('actual_owned_worker_pid')==os.getpid():
            lease.update(running=False,actual_GPU_running=False,actual_training_optimizer_running=False,actual_owned_worker_pid=None,
                status='production_resident_RAM_process_closed',committed_updates=locals().get('last_checkpoint',initial_updates));write_json(lease_path,lease)
""", """    finally:
        try:close_private_metadata()
        finally:
            context_safe=prefetch is None or(not prefetch.thread.is_alive() and prefetch.worker_status['closed'])
            with ExitStack() as cleanup:
                cleanup.callback(release_current_lease,locals().get('last_checkpoint',initial_updates))
                cleanup.callback(monitor.close)
                if engine is not None and context_safe:cleanup.callback(engine.close)
                elif engine is not None:
                    write_json(invocation/'context_close_deferred_to_abnormal_process_exit.json',
                        {'reason':'Private worker did not join; never close its continuously locked source authority early',
                         'normal_context_or_worker_closure_claimed':False})
                cleanup.callback(scopes.close)
                if stream is not None:cleanup.callback(stream.close)
""")
ast.parse(s)
new_driver.write_bytes(s.replace('\n','\r\n').encode('utf-8'))
w=old_wrapper.read_text(encoding='utf-8')
w=replace(w,'run_ASR18_BPE9223_resident_RAM_training_v4.py','run_ASR18_BPE9223_resident_RAM_training_v5.py')
w=replace(w,"""    require(a.queue_bytes==review['bounded_RAM_queue_bytes']==268435456 and review['periodic_DEV_interval_updates']==4000
        and a.DEV_frontend_resume is not None and review['DEV_frontend_resume']==record(a.DEV_frontend_resume),'Exact changed queue/periodic/abnormal source binding')
""", """    require(a.queue_bytes==review['bounded_RAM_queue_bytes']==268435456 and review['periodic_DEV_interval_updates']==4000,
        'Unchanged bounded queue and periodic cadence')
    configuration=read_json(a.configuration);policy=configuration['private_scalar_prefetch_policy']
    require(review['private_scalar_prefetch_source']==policy['source']==record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py')
        and review['private_scalar_prefetch_actual_CPU_review']==policy['actual_CPU_control_review']
        and record(policy['actual_CPU_control_review']['path'])==policy['actual_CPU_control_review']
        and review['additional_prefetch_host_reservation_bytes']==policy['additional_host_reservation_bytes']==268435456,
        'Exact integrated helper, finite equality review and full added host reservation')
    if a.prior_process is not None:
        require(a.DEV_frontend_resume is None,'New normal-boundary entry requires genuine prior wait0, not historical failure relabel')
    else:
        require(a.DEV_frontend_resume is not None and review['DEV_frontend_resume']==record(a.DEV_frontend_resume),
            'Only the already-supported typed4000 failure lane when no genuine prior process0')
""")
ast.parse(w)
new_wrapper.write_bytes(w.replace('\n','\r\n').encode('utf-8'))
base=N/'resident_RAM_DEV_frontend_source_v1/configuration.json'
c=json.loads(base.read_text(encoding='utf-8'))
c['private_scalar_prefetch_policy']={'kind':'ASR18_private_scalar_row_prefetch_policy_v1',
    'source':pin(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
    'actual_CPU_control_review':pin(N/'ROOT_private_scalar_prefetch_actual_CPU_control_review_v1.json'),
    'additional_host_reservation_bytes':268435456,'maximum_TR_rows':280,'maximum_canonical_row_bytes':8388608,
    'SQLite_cache_bytes_each':4194304,'supported_phase':'weak_label_continued_training',
    'original_MainThread_planning_and_commits_unchanged':True,'idle_during_DEV':True}
config=O/'configuration.json';assert not config.exists();config.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
for old,new,name in ((old_driver,new_driver,'driver4_to5.diff.txt'),(old_wrapper,new_wrapper,'wrapper4_to5.diff.txt')):
    diff=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),new.read_text(encoding='utf-8').splitlines(True),fromfile=str(old),tofile=str(new)))
    (O/name).write_text(diff,encoding='utf-8')
print(json.dumps({'driver':pin(new_driver),'wrapper':pin(new_wrapper),'configuration':pin(config),'status':'INACTIVE_NOT_DEPLOYED'}))
