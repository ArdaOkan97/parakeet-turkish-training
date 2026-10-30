"""Derive inactive18 from frozen17; no numerical or resource change."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';D=Path(__file__).parent;N=D.parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');F=N/'resident_RAM_periodic_quiescence140000_failure_actual_v1'
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def req(ok,message):
 if not ok:raise ValueError(message)
def write(path,text):req(not path.exists(),'Preserve '+str(path));path.write_text(text,encoding='utf-8')
old=(R/'run_ASR18_BPE9223_resident_RAM_training_v17.py').read_text(encoding='utf-8-sig');new=old;changes=[]
def rep(a,b,label,count=1):
 global new
 req(new.count(a)==count,label+' count '+str(new.count(a)));new=new.replace(a,b);changes.append({'label':label,'old':a,'new':b,'count':count})
gate='''def checked_periodic_quiescence_resume(pin,*,base,folder,pointer_pin):
    import psutil
    require(pin==base['prior_resident_RAM_periodic_quiescence_failure'] and record(pin['path'])==pin
        and pin['sha256']=='3c5d30512e62a77c846ec5ad570f88addea46cfc58498b2caa38e05274596ac9',
        'Exact actual140000 private-plan equality failure, never normal0')
    value=read_json(pin['path']);inputs=value['inputs'];failure=value['actual_failure']
    require(value['kind']=='ASR18_actual_saved140000_periodic_CPU_quiescence_plan_identity_failure_closure_v1'
        and value['owned_session']==86052 and value['actual_outer_TOOL_exit_code']==1
        and value['all_committed_and_fully_saved_updates']==140000 and value['unsaved_positive_tail_updates']==0
        and value['pending140000_DEV_not_completed'] is True and value['pending140000_DEV_must_precede140001'] is True
        and value['native_numeric_OS_exit_code'] is None and value['CPU_numeric_OS_exit_code'] is None
        and value['real_periodic_CPU_quiescence_wait0_not_claimed'] is True,
        'Preserve actual TOOL1 and uncompleted140k DEV/CPUclose, no numeric exit invention')
    for evidence in inputs.values():require(record(evidence['path'])==evidence,'Exact compact140000 failure evidence')
    require(failure==read_json(inputs['actual_failure']['path'])
        and failure['error']=="ValueError('Keep the exact previously submitted private plan')"
        and failure['actual_optimizer_updates']==failure['last_checkpoint']==140000
        and 'close_private_preparation_for_DEV' in failure['traceback'],
        'Actual original helper1 failed before any packet drain, not a memory or EOF relabel')
    expected=[('native',41200,1792660196.3122573),('native_interpreter_launcher',38908,1792660196.3051417),
        ('wrapper',31740,None),('CPU',26908,1792660472.4757633),('CPU_interpreter_launcher',33564,None)]
    require([(x['role'],x['pid'],x['recorded_create_time'])for x in value['all_recorded_identities_dead']]==expected,
        'Exact140000 predecessor identities and recorded timestamp limitations')
    for role,pid,ctime in expected:
        try:current=psutil.Process(pid).create_time();dead=ctime is not None and abs(current-ctime)>.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Recorded140000 predecessor actually dead '+role)
    require(pointer_pin==inputs['pointer'] and record(pointer_pin['path'])==pointer_pin,'Exact fully saved140000 pointer')
    pointer=read_json(pointer_pin['path']);saved=pointer['journal_prefix'];whole=value['whole_journal_verified_once']
    require(pointer['committed_updates']==140000 and saved=={'path':str(folder/'journals/commits_000012.jsonl'),
        'offset':284584907,'sha256':'caeb1f6fe40c867028bff40b0aa7f838a7a9d185872855213adc4d2a75d697af'}
        and whole=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']}
        and Path(saved['path']).stat().st_size==saved['offset'],'Same whole saved140000 journal, no unsaved tail/copy/truncation')
    persistence=read_json(inputs['persistence']['path']);monitor=read_json(inputs['monitor_completion']['path'])
    require(persistence['passed'] and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['journal_prefix']==saved
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and monitor['joined'] and monitor['clean'] and monitor['error'] is None,'Actual all-state saved metadata and clean monitor')
    current=read_json(inputs['last_consumed_plan']['path']);future=read_json(inputs['next_unconsumed_plan']['path'])
    require(current['initial_committed_updates']==139980 and future['initial_committed_updates']==140000
        and len(current['updates'])==len(future['updates'])==20,'Exact current200 and unconsumed successor201 provenance')
    return {'saved_journal_prefix':saved,'whole_journal':whole,'actual_failure':pin,
        'pending_periodic_DEV_updates':140000,'unsaved_tail_bytes':0}


'''
rep('def private_successor_request(',gate+'def private_successor_request(','new exact140000 abnormal gate')
rep('asr18_resident_periodic_CPU_quiescence_v1','asr18_resident_periodic_CPU_quiescence_v2','exact helper2 import/policy pin',2)
rep("    require(sum(x is not None for x in (abnormal,frontend_failure,transition,interval_failure,DEV_admission_failure))<=1,'One truthful predecessor lane')", "    periodic_quiescence_failure=None if a.periodic_quiescence_resume is None else checked_periodic_quiescence_resume(\n        record(a.periodic_quiescence_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    require(sum(x is not None for x in (abnormal,frontend_failure,transition,interval_failure,DEV_admission_failure,periodic_quiescence_failure))<=1,'One truthful predecessor lane')",'one actual predecessor')
rep('or interval_failure is not None or DEV_admission_failure is not None),','or interval_failure is not None or DEV_admission_failure is not None or periodic_quiescence_failure is not None),','existing branch supported typed failure')
rep('and interval_failure is None and DEV_admission_failure is None:', 'and interval_failure is None and DEV_admission_failure is None and periodic_quiescence_failure is None:','normal0 remains strict')
rep("        elif DEV_admission_failure is not None:require(a.prior_process is None and a.invocation==13,'Actual DEV prospective admission TOOL1 is never normal0')", "        elif DEV_admission_failure is not None:require(a.prior_process is None and a.invocation==13,'Actual DEV prospective admission TOOL1 is never normal0')\n        elif periodic_quiescence_failure is not None:require(a.prior_process is None and a.invocation==14,'Actual helper1 equality failure is never normal0')",'exact initial14 only')
rep('            if DEV_admission_failure is not None:\n', "            if periodic_quiescence_failure is not None:\n                require(saved==source_journal and record(saved['path'])==periodic_quiescence_failure['whole_journal'],'Exact whole saved140000 journal')\n                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved140000_prefix_restore_after_periodic_quiescence_failure_v1',\n                    'failure':record(a.periodic_quiescence_resume),'original_tail_modified':False,'no_tail_replay':True}\n            if DEV_admission_failure is not None:\n",'same genuine saved-prefix restore ABI')
rep('or interval_failure is not None or DEV_admission_failure is not None\n','or interval_failure is not None or DEV_admission_failure is not None or periodic_quiescence_failure is not None\n','fresh restore whole view')
pending="""        if periodic_quiescence_failure is not None:
            require(optimizer.updates==140000 and 140000 not in evaluated and queued is None,
                'Pending failed140000 DEV before any producer submission or140001')
            lease.update(status='production_resident_RAM_serial350_DEV',actual_training_optimizer_running=False);write_json(lease_path,lease)
            restored_status.update(phase='pending_periodic_DEV_140000_before140001');write_json(folder/'progress/latest.json',restored_status)
            periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
            dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_000140000_attempt_'+str(a.invocation).zfill(6)))
            record_periodic(dev_pin,score,preservation,False)
            require(optimizer.updates==140000 and journal.binding()==source_journal,
                'Pending140000 DEV completed before positive Adam, draws or new journal bytes')
            lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
"""
rep("        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)",pending+"        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)",'pending140k before first CPU/native')
rep("    p.add_argument('--DEV-admission-resume');", "    p.add_argument('--periodic-quiescence-resume');p.add_argument('--DEV-admission-resume');",'new explicit flag')
ast.parse(new);inverse=new
for c in reversed(changes):req(inverse.count(c['new'])==c['count'],'Inverse '+c['label']);inverse=inverse.replace(c['new'],c['old'])
req(inverse==old,'Full driver inverse equality');source=R/'run_ASR18_BPE9223_resident_RAM_training_v18.py';write(source,new)
write(D/'driver17_to18.diff.txt',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='driver17',tofile='driver18')));write(D/'driver18_declared_replacements.json',json.dumps(changes,indent=2)+'\n')
base_path=N/'resident_RAM_periodic_DEV_quiescence_source_v1/configuration_candidate.json';base=json.loads(base_path.read_text());config=json.loads(base_path.read_text());config['periodic_DEV_CPU_quiescence_policy']['source']=pin(S/'asr18_resident_periodic_CPU_quiescence_v2.py');config['prior_resident_RAM_periodic_quiescence_failure']=pin(F/'actual_failure_closure.json');check=json.loads(json.dumps(config));check.pop('prior_resident_RAM_periodic_quiescence_failure');check['periodic_DEV_CPU_quiescence_policy']['source']=base['periodic_DEV_CPU_quiescence_policy']['source'];req(check==base,'Only operational helper source and actual failure')
write(D/'configuration_candidate.json',json.dumps(config,indent=2)+'\n')
write(D/'driver18_inactive_handoff.json',json.dumps({'kind':'ASR18_inactive_saved140000_periodic_quiescence_representation_driver18_v1','driver_source':pin(source),'helper_source':pin(S/'asr18_resident_periodic_CPU_quiescence_v2.py'),'driver_diff':pin(D/'driver17_to18.diff.txt'),'driver_declared_replacements':pin(D/'driver18_declared_replacements.json'),'helper_diff':pin(D/'helper1_to2.diff.txt'),'configuration':pin(D/'configuration_candidate.json'),'base_configuration':pin(base_path),'actual_failure':pin(F/'actual_failure_closure.json'),'full_inverse_source_equality':True,'only_two_operational_top_level_fields_changed':True,'CPU7_native_DEV_prefetch_and_resource_numbers_unchanged':True,'actual_control_executed':False,'GPU_launch_authorized':False},indent=2)+'\n');print(json.dumps(pin(D/'driver18_inactive_handoff.json')))
