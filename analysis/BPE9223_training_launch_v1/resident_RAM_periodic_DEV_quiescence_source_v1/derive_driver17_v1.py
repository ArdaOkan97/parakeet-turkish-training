"""Derive an inactive source17 phase-scheduling successor; preserve frozen16."""
import ast, difflib, hashlib, json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20'); S=P/'outputs/parakeet_turkish'; R=P/'work/research'; D=Path(__file__).parent
F=D.parent/'resident_RAM_DEV_admission136000_failure_actual_v1'; B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def req(ok,msg):
 if not ok:raise ValueError(msg)
def emit(path,text):
 req(not path.exists(),'Preserve '+str(path));path.write_text(text,encoding='utf-8')
old=(R/'run_ASR18_BPE9223_resident_RAM_training_v16.py').read_text(encoding='utf-8-sig'); new=old;changes=[]
def replace(a,b,label,count=1):
 global new
 req(new.count(a)==count,label+' exact occurrence '+str(new.count(a)));new=new.replace(a,b);changes.append({'label':label,'old':a,'new':b,'count':count})
gate='''def checked_DEV_admission_resume(pin,*,base,folder,pointer_pin):
    import psutil
    require(pin==base['prior_resident_RAM_DEV_admission_failure'] and record(pin['path'])==pin
        and pin['sha256']=='89788d67c857c84c74216f90eef72c891a18eb7a216514a68eecf0e4872bee21',
        'Exact actual136000 DEV prospective admission failure, not normal0')
    value=read_json(pin['path']);inputs=value['inputs'];failure=value['actual_failure']
    require(value['kind']=='ASR18_actual_saved136000_resident_DEV_prospective_admission_failure_closure_v1'
        and value['owned_session']==64681 and value['actual_outer_TOOL_exit_code']==1
        and value['actual_committed_and_fully_saved_updates']==136000 and value['unsaved_positive_tail_updates']==0
        and value['pending136000_DEV_not_completed'] is True and value['pending136000_DEV_must_precede136001'] is True
        and value['native_numeric_OS_exit_code'] is None and value['CPU_numeric_OS_exit_code'] is None,
        'Truthful abnormal wait1 with unknown numeric native exit and no136k DEV result')
    for evidence in inputs.values():require(record(evidence['path'])==evidence,'Exact compact actual136000 evidence')
    require(failure==read_json(inputs['actual_failure']['path'])
        and failure['error']=="ValueError('Actual allocation increment plus full extra prefetch reservation and unchanged commit headroom')"
        and failure['actual_optimizer_updates']==failure['last_checkpoint']==136000
        and "admit('resident_optimizer_idle_serial350_DEV',2100000000)" in failure['traceback'],
        'Exact unchanged prospective DEV guard, not an audio or normal stop relabel')
    expected=[('native',27232,1792655703.93878),('native_interpreter_launcher',25192,1792655703.9323223),
        ('wrapper',35268,None),('CPU',548,1792655894.7857172),('CPU_interpreter_launcher',35232,None)]
    require([(v['role'],v['pid'],v['recorded_create_time'])for v in value['all_recorded_identities_dead']]==expected,
        'Exact old identities; never invent historical create-times')
    for role,pid,ctime in expected:
        try:current=psutil.Process(pid).create_time();dead=ctime is not None and abs(current-ctime)>.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Actual recorded136000 predecessor identity dead '+role)
    require(pointer_pin==inputs['pointer'] and record(pointer_pin['path'])==pointer_pin,'Exact fully verified saved136000 pointer')
    pointer=read_json(pointer_pin['path']);saved=pointer['journal_prefix'];whole=value['whole_journal_verified_once']
    require(pointer['committed_updates']==136000 and saved=={'path':str(folder/'journals/commits_000012.jsonl'),
        'offset':276724922,'sha256':'40bce7c886d3e053543e6bd0c4ec6c111c5b6f29aab13dcc1beab50443010b6e'}
        and whole=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']}
        and Path(saved['path']).stat().st_size==saved['offset'],'Exact whole saved lineage; no tail, truncation or branch copy')
    persistence=read_json(inputs['persistence']['path']);monitor=read_json(inputs['monitor_completion']['path'])
    require(persistence['passed'] and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['journal_prefix']==saved
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,
        'Actual saved all-state metadata and joined clean monitor')
    return {'saved_journal_prefix':saved,'whole_journal':whole,'actual_failure':pin,
        'pending_periodic_DEV_updates':136000,'unsaved_tail_bytes':0}


'''
replace('def private_successor_request(',gate+'def private_successor_request(','new exact failure gate')
policy="""    from asr18_resident_periodic_CPU_quiescence_v1 import close_private_preparation_for_DEV
    require(base['periodic_DEV_CPU_quiescence_policy']=={
        'kind':'ASR18_periodic_DEV_private_CPU_quiescence_policy_v1',
        'source':record(S/'asr18_resident_periodic_CPU_quiescence_v1.py'),
        'whole_unconsumed_private_successor_only':True,'same_plan_resubmitted_after_DEV':True,
        'CPU_close_requires_actual_wait0_and_joined_handles':True,
        'DEV_incremental_allowance_bytes':2100000000,'additional_prefetch_reservation_bytes':268435456,
        'commit_headroom_bytes':2000000000,'physical_headroom_bytes':1500000000},
        'Only changed phase scheduling; preserve every allocation and actual headroom guard')
"""
replace('    abnormal=None if a.abnormal_resume is None else checked_headroom_resume(',policy+'    abnormal=None if a.abnormal_resume is None else checked_headroom_resume(','exact phase policy')
replace("    require(sum(x is not None for x in (abnormal,frontend_failure,transition,interval_failure))<=1,'One truthful predecessor lane')", "    DEV_admission_failure=None if a.DEV_admission_resume is None else checked_DEV_admission_resume(\n        record(a.DEV_admission_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    require(sum(x is not None for x in (abnormal,frontend_failure,transition,interval_failure,DEV_admission_failure))<=1,'One truthful predecessor lane')",'one exact typed lane')
replace('or transition is not None or interval_failure is not None),','or transition is not None or interval_failure is not None or DEV_admission_failure is not None),','existing branch typed entry')
replace('if abnormal is None and frontend_failure is None and transition is None and interval_failure is None:', 'if abnormal is None and frontend_failure is None and transition is None and interval_failure is None and DEV_admission_failure is None:','normal0 unchanged strictness')
replace("        elif interval_failure is not None:require(a.prior_process is None and a.invocation==12,'Actual original-interval TOOL1 is not normal0/intentional stop')", "        elif interval_failure is not None:require(a.prior_process is None and a.invocation==12,'Actual original-interval TOOL1 is not normal0/intentional stop')\n        elif DEV_admission_failure is not None:require(a.prior_process is None and a.invocation==13,'Actual DEV prospective admission TOOL1 is never normal0')",'typed initial13 only')
replace("            require(s['system_memory']['commit_available_bytes']-increment-extra>=2000000000,", "            if s['system_memory']['commit_available_bytes']-increment-extra<2000000000:\n                write_json(invocation/'actual_prospective_admission_denial.json',{'name':name,'sample':s,\n                    'increment_bytes':increment,'additional_prefetch_reservation_bytes':extra,\n                    'required_remaining_commit_bytes':2000000000,'no_allocation_admitted':True})\n            require(s['system_memory']['commit_available_bytes']-increment-extra>=2000000000,",'persist denied actual sample before unchanged guard')
replace('            if frontend_failure is not None:\n', "            if DEV_admission_failure is not None:\n                require(saved==source_journal and record(saved['path'])==DEV_admission_failure['whole_journal'],'Exact whole saved136000 journal')\n                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved136000_prefix_restore_after_DEV_admission_failure_v1',\n                    'failure':record(a.DEV_admission_resume),'original_tail_modified':False,'no_tail_replay':True}\n            if frontend_failure is not None:\n",'typed actual saved-prefix restore callback')
replace('or transition is not None or interval_failure is not None\n', 'or transition is not None or interval_failure is not None or DEV_admission_failure is not None\n','same genuine fresh restore view')
pending="""        if DEV_admission_failure is not None:
            require(optimizer.updates==136000 and 136000 not in evaluated and queued is None,
                'Pending failed136000 DEV before any producer submission or136001')
            lease.update(status='production_resident_RAM_serial350_DEV',actual_training_optimizer_running=False);write_json(lease_path,lease)
            restored_status.update(phase='pending_periodic_DEV_136000_before136001');write_json(folder/'progress/latest.json',restored_status)
            periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
            dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_000136000_attempt_'+str(a.invocation).zfill(6)))
            record_periodic(dev_pin,score,preservation,False)
            require(optimizer.updates==136000 and journal.binding()==source_journal,
                'Pending136000 DEV completed before positive Adam, draws or new journal bytes')
            lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
"""
replace("        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)",pending+"        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)",'evaluate pending136000 before first producer')
replace("            if periodic_due:\n                periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)", "            if periodic_due:\n                CPU_quiescence=close_private_preparation_for_DEV(stream,queued,committed_updates=optimizer.updates,guard=guard)\n                stream=None\n                require_exact_live_restore(before,live_state_fingerprint(optimizer,torch=torch,guard=guard))\n                require(journal.binding()==ptr['journal_prefix'] and digest(engine.schedule.state_dict())==ptr['sampler_state_sha256'],\n                    'Private CPU closure leaves saved weights/threeAdam/fourRNG/both cursors/journal unchanged')\n                write_json(invocation/('CPU_periodic_'+str(optimizer.updates).zfill(9)+'_quiescence.json'),CPU_quiescence)\n                periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)",'genuine CPU close before periodic DEV')
replace("            require(queued is not None,'Exact private successor after periodic or normal bounded window')\n            plan,plan_pin=queued", "            require(queued is not None,'Exact private successor after periodic or normal bounded window')\n            if periodic_due:\n                stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/('CPU_after_periodic_DEV_'+str(optimizer.updates).zfill(9)),queue_bytes=a.queue_bytes,guard=guard)\n                stream.submit(queued[1]) # Same immutable private plan; no sampler/plan regeneration.\n            plan,plan_pin=queued",'same exact private successor resubmission after DEV')
replace("    p.add_argument('--original-interval-resume');", "    p.add_argument('--DEV-admission-resume');p.add_argument('--original-interval-resume');",'new explicit CLI predecessor')
ast.parse(new)
reconstructed=new
for change in reversed(changes):
 req(reconstructed.count(change['new'])==change['count'],'Inverse '+change['label']);reconstructed=reconstructed.replace(change['new'],change['old'])
req(reconstructed==old,'Exact full-source inverse equality')
source=R/'run_ASR18_BPE9223_resident_RAM_training_v17.py';emit(source,new)
emit(D/'driver16_to17.diff.txt',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='driver16',tofile='driver17')))
emit(D/'driver17_declared_replacements.json',json.dumps(changes,indent=2)+'\n')
base=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1\resident_RAM_original_interval_post134529_failure_actual_v1\production_configuration_v1\configuration.json'); configuration=json.loads(base.read_text())
configuration['prior_resident_RAM_DEV_admission_failure']=pin(F/'actual_failure_closure.json')
configuration['periodic_DEV_CPU_quiescence_policy']={'kind':'ASR18_periodic_DEV_private_CPU_quiescence_policy_v1','source':pin(S/'asr18_resident_periodic_CPU_quiescence_v1.py'),'whole_unconsumed_private_successor_only':True,'same_plan_resubmitted_after_DEV':True,'CPU_close_requires_actual_wait0_and_joined_handles':True,'DEV_incremental_allowance_bytes':2100000000,'additional_prefetch_reservation_bytes':268435456,'commit_headroom_bytes':2000000000,'physical_headroom_bytes':1500000000}
check=dict(configuration);check.pop('prior_resident_RAM_DEV_admission_failure');check.pop('periodic_DEV_CPU_quiescence_policy');req(check==json.loads(base.read_text()),'Only two added operational fields')
emit(D/'configuration_candidate.json',json.dumps(configuration,indent=2)+'\n')
emit(D/'driver17_inactive_handoff.json',json.dumps({'kind':'ASR18_inactive_periodic_DEV_CPU_quiescence_driver17_source_v1','driver':pin(source),'helper':pin(S/'asr18_resident_periodic_CPU_quiescence_v1.py'),'diff':pin(D/'driver16_to17.diff.txt'),'declared_replacements':pin(D/'driver17_declared_replacements.json'),'configuration':pin(D/'configuration_candidate.json'),'base_configuration':pin(base),'actual_failure':pin(F/'actual_failure_closure.json'),'CPU_source_unchanged':pin(S/'fastmss_resident_feature_stream_v7.py'),'full_inverse_source_equality':True,'configuration_exact_after_two_added_operational_fields':True,'active_deployment':False,'CPU_controls_executed':False},indent=2)+'\n')
print(json.dumps(pin(D/'driver17_inactive_handoff.json')))
