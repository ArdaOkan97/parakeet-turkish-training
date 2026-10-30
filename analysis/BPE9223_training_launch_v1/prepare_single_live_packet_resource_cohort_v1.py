"""Inactive scheduling-only successor; no worker/model/audio launch."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(__file__).parent;D=N/'resident_RAM_single_packet_resource_source_v1'
def require(ok,message):
    if not ok:raise ValueError(message)
def pin(p):
    p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
changes=[]
def replace(text,old,new,count=1):
    require(text.count(old)==count,'Exact declared source occurrence: '+old[:90]);changes.append(dict(old=old,new=new,count=count));return text.replace(old,new)
profile_body='''def checked_resource_profile(config):
    expected={'kind':'ASR18_resident_RAM_single_live_packet_resource_scheduling_v1',
        'CPU_protocol_source':pin(__file__),'maximum_live_receiver_packets':1,
        'queue_capacity_bytes':268435456,'maximum_packet_bytes':268435456,
        'CPU_prospective_commit_reserve_bytes':2500000000,'CPU_allocation_wait_maximum_seconds':300,
        'CPU_actual_commit_headroom_bytes':1500000000,'native_actual_commit_headroom_bytes':2000000000,
        'native_actual_physical_headroom_bytes':1500000000,'CPU_private_limit_bytes':8000000000,
        'additional_prefetch_host_reservation_bytes':268435456,'freed_reservation_credit_claimed':False}
    require(config.get('resident_RAM_resource_scheduling_policy')==expected,
            'Exact one-live-packet workload and stronger prospective CPU reserve; original guards retained')
    return expected


'''
old_cpu=S/'fastmss_resident_feature_stream_v5.py';new_cpu=S/'fastmss_resident_feature_stream_v6.py'
c=old_cpu.read_text(encoding='utf-8');cpu_old=c
c=replace(c,'class AvailableMemoryPolicy:',profile_body+'class AvailableMemoryPolicy:')
c=replace(c,'        self.peak_commit_fraction = 0.\n','        self.peak_commit_fraction = 0.\n        self.prospective_reserve_bytes = 2_500_000_000\n        self.allocation_wait_seconds = 300.\n        self.allocation_wait_count = 0\n        self.allocation_wait_seconds_total = 0.\n        self.last_observed = None\n')
c=replace(c,"                'Fresh actual private/system memory telemetry')\n", "                'Fresh actual private/system memory telemetry')\n        self.last_observed = value\n")
old_admit=c[c.index('    def admit(self, budget, guard=None):'):c.index('\n\ndef _raw_finish_without_full_heap_collection')]
new_admit='''    def admit(self, budget, guard=None):
        require(type(budget) is int and 0 < budget <= self.private_limit, 'Explicit imminent allocation budget')
        began=time.monotonic();waited=False
        while True:
            value=(guard or self)();system=value['system_memory']
            require(value['worker_private_memory']['private_bytes']+budget<=self.private_limit,
                    'Current resident private plus imminent allocation exceeds budget')
            if system['commit_available_bytes']-budget>=self.prospective_reserve_bytes:break
            if not waited:self.allocation_wait_count+=1;waited=True
            elapsed=time.monotonic()-began
            if elapsed>=self.allocation_wait_seconds:
                self.allocation_wait_seconds_total+=elapsed
                raise TimeoutError('Bounded prospective CPU allocation headroom wait exhausted')
            time.sleep(.1)
        elapsed=time.monotonic()-began
        if waited:self.allocation_wait_seconds_total+=elapsed
        return {'sample':value,'forecast_incremental_bytes':budget,'passed':True,
                'policy':'single_live_packet_stronger_prospective_CPU_reserve_v1',
                'prospective_commit_reserve_bytes':self.prospective_reserve_bytes,
                'allocation_wait_seconds':elapsed if waited else 0.}
'''
c=replace(c,old_admit,new_admit)
c=replace(c,'    policy = AvailableMemoryPolicy(private_limit_bytes)\n',"    profile=checked_resource_profile(bound(configuration_path))\n    require(private_limit_bytes==profile['CPU_private_limit_bytes'],'Same explicit CPU private budget')\n    policy = AvailableMemoryPolicy(private_limit_bytes)\n")
c=replace(c,"                       'peak_commit_fraction': policy.peak_commit_fraction})", "                       'peak_commit_fraction': policy.peak_commit_fraction,\n                       'allocation_wait_count':policy.allocation_wait_count,\n                       'allocation_wait_seconds':policy.allocation_wait_seconds_total,\n                       'prospective_commit_reserve_bytes':policy.prospective_reserve_bytes})")
c=replace(c,"            _send(output, {'kind': 'error', 'message': repr(exc), 'completed_updates': completed})", "            _send(output, {'kind': 'error', 'message': repr(exc), 'completed_updates': completed,\n                'actual_resource_telemetry':{'last_observed_memory':policy.last_observed,\n                    'peak_private_bytes':policy.peak_private,'allocation_wait_count':policy.allocation_wait_count,\n                    'allocation_wait_seconds':policy.allocation_wait_seconds_total,\n                    'prospective_commit_reserve_bytes':policy.prospective_reserve_bytes}})")
c=replace(c,"        config = bound(configuration_pin)\n", "        config = bound(configuration_pin)\n        profile=checked_resource_profile(config)\n        require(queue_bytes==profile['queue_capacity_bytes'] and private_limit_bytes==profile['CPU_private_limit_bytes'],\n                'Unchanged full packet/queue byte and CPU private reservations')\n        self.max_live_packets=profile['maximum_live_receiver_packets']\n        self.peak_live_packets=0;self.credit_wait_seconds=0.\n")
c=replace(c,"                command = self.commands.get()\n                _send(self.process.stdin, command)", "                command = self.commands.get()\n                if command['kind']=='abort_sender':return\n                _send(self.process.stdin, command)")
c=replace(c,"                        while self.used_bytes + count > self.queue_bytes and self.error is None:\n                            self.condition.wait(.25)", "                        credit_began=time.monotonic()\n                        while (self.used_bytes+count>self.queue_bytes or len(self.live)>=self.max_live_packets) and self.error is None:\n                            self.condition.wait(.25)\n                        self.credit_wait_seconds+=time.monotonic()-credit_began")
c=replace(c,"                        self.live[packet['token']] = {'bytes': count, 'delivered': False}\n", "                        self.live[packet['token']] = {'bytes': count, 'delivered': False}\n                        self.peak_live_packets=max(self.peak_live_packets,len(self.live))\n")
c=replace(c,"                elif kind == 'error':\n                    raise RuntimeError", "                elif kind == 'error':\n                    (self.folder/'actual_CPU_failure.json').write_bytes(canonical(frame))\n                    raise RuntimeError")
c=replace(c,"        self._sender.join(timeout=10)\n        self.log.close()", "        self._sender.join(timeout=10)\n        require(not self._receiver.is_alive() and not self._sender.is_alive(),'Actual resident CPU transport threads joined')\n        self.process.stdin.close();self.process.stdout.close()\n        self.log.close()")
c=replace(c,"                  'queue_capacity_bytes': self.queue_bytes, 'peak_queue_bytes': self.peak_queue_bytes}", "                  'queue_capacity_bytes': self.queue_bytes, 'peak_queue_bytes': self.peak_queue_bytes,\n                  'maximum_live_receiver_packets':self.max_live_packets,'peak_live_receiver_packets':self.peak_live_packets,\n                  'receiver_credit_wait_seconds':self.credit_wait_seconds,'transport_threads_joined':True,'pipe_handles_closed':True}")
c=replace(c,"        self._fail(RuntimeError('Resident producer explicitly aborted; no normal exit claimed'))\n", "        self._fail(RuntimeError('Resident producer explicitly aborted; no normal exit claimed'))\n        try:self.commands.put_nowait({'kind':'abort_sender'})\n        except queue.Full:pass\n")
c=replace(c,"            self.process.wait(timeout=30)\n        self.log.close()", "            self.process.wait(timeout=30)\n        self._receiver.join(timeout=10);self._sender.join(timeout=10)\n        require(not self._receiver.is_alive() and not self._sender.is_alive(),'Aborted transport threads genuinely joined')\n        self.process.stdin.close();self.process.stdout.close()\n        self.log.close()")
cpu_changes=copy.deepcopy(changes);changes=[]

old_driver=R/'run_ASR18_BPE9223_resident_RAM_training_v12.py';new_driver=R/'run_ASR18_BPE9223_resident_RAM_training_v15.py'
d=old_driver.read_text(encoding='utf-8');driver_old=d
old_gate=d[d.index('def checked_headroom_resume('):d.index('\n\ndef checked_DEV_frontend_resume(')]
new_gate='''def checked_headroom_resume(pin,*,base,folder,pointer_pin):
    import psutil
    require(pin==base['prior_resident_RAM_headroom_failure'] and record(pin['path'])==pin,'Exact ROOT-bound actual133137 resource failure')
    value=read_json(pin['path']);e=value['source_evidence']
    require(value['kind']=='ASR18_actual_resident_RAM_headroom133137_saved133000_abnormal_closure_v1'
        and value['actual_TOOL_exit_code']==1 and value['actual_TOOL_chunk_id']=='d0a49f'
        and value['actual_failed_optimizer_updates']==133137 and value['saved_optimizer_updates']==133000
        and value['normal_completion_claimed'] is False and value['original_failure_was_resource_guard_not_normal0'] is True
        and value['native_OS_exit_code'] is None and value['native_OS_exit_code_not_observed'] is True,
        'Truthful abnormal stop, unknown numeric native exit and exact saved133000')
    require(e['saved_pointer']==pointer_pin and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
        and e['driver_source']==record(R/'run_ASR18_BPE9223_resident_RAM_training_v12.py')
        and e['wrapper_source']==record(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v12.py'),
        'Exact original source12 failure and preserved branch checkpoint')
    for item in e.values():require(record(item['path'])==item,'Exact original failure/source/checkpoint evidence')
    require(record(value['owned_process_deaths']['path'])==value['owned_process_deaths'],'Exact actual owned-death observation')
    observed=read_json(value['owned_process_deaths']['path'])['identities']
    require(len(observed)==5 and {x['label']for x in observed}=={'native','native interpreter launcher','resident CPU','CPU interpreter launcher','outer wrapper'},'Only the five actually recorded identities')
    for item in observed:
        who=item['original_identity']
        try:current=psutil.Process(who['pid']).create_time()
        except psutil.NoSuchProcess:continue
        require(who['create_time'] is not None and abs(current-who['create_time'])>=.001,'Original owned identity is dead; never invent missing create time')
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path'])
    sample=read_json(e['failing_headroom_sample']['path'])
    require(failure['actual_optimizer_updates']==133137 and failure['last_checkpoint']==133000
        and 'Sensible real system headroom' in failure['error'] and monitor['joined'] is True and monitor['clean'] is False
        and sample['system_memory']['commit_available_bytes']==1639993344
        and sample['system_memory']['physical_available_bytes']==12542930944,
        'Genuine captured system-commit guard failure, not normal0 or an interval repair')
    saved={k:value['saved_journal_prefix'][k]for k in ('path','offset','sha256')}
    whole=value['whole_old_journal'];tail=value['preserved_unsaved_positive_tail']
    require(saved=={'path':str(folder/'journals/commits_000010.jsonl'),'offset':270832675,
        'sha256':'a2811751b95fd8c6df021d3495b4af77e0ccd315134e667a09e893da97dfdc0e'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved
        and whole=={'path':saved['path'],'bytes':271101236,'sha256':'99014a7440de7360f3c1065b09029604e8c0e8a6025e84ba6872a7398876abd1'}
        and Path(whole['path']).stat().st_size==whole['bytes'],'Exact saved prefix and preserved whole old lineage')
    tail_pin={k:tail[k]for k in ('path','bytes','sha256')}
    require(tail_pin['bytes']==268561 and tail_pin['sha256']=='227764ba62b5fee6fa87aaeabf9d4f0197ea2b414f41556d2a60310afb24dc8c'
        and record(tail_pin['path'])==tail_pin,'Exact137 historical positive tail, never replayed')
    events=[__import__('json').loads(line)for line in Path(tail_pin['path']).read_bytes().splitlines()]
    require(len(events)==137 and all(x['optimizer_updates']==u and x['native']['update']==u
        and x['prepared_evidence']['actual_positive_native_update']==u for u,x in zip(range(133001,133138),events)),
        'Actual ordered137 positive events remain historical')
    return {**value,'saved_journal_prefix':saved,'whole_journal_including_preserved_tail':whole,
        'preserved_unsaved_tail':tail_pin,'preserved_tail_positive_updates':list(range(133001,133138))}
'''
# R is not a global in the driver; use its existing P/work/research authority.
new_gate=new_gate.replace("record(R/'", "record(P/'work/research/")
d=replace(d,old_gate,new_gate)
d=replace(d,"    require(a.queue_bytes==268435456,'Changed bounded256MiB queue retains every full admissible256MiB packet')", "    require(a.queue_bytes==268435456,'Unchanged256MiB queue retains every full admissible256MiB packet')\n    from fastmss_resident_feature_stream_v6 import checked_resource_profile\n    checked_resource_profile(base)")
d=replace(d,"elif abnormal is not None:require(a.prior_process is None and a.invocation==4,'Resource failure is not a normal0 predecessor')", "elif abnormal is not None:require(a.prior_process is None and a.invocation==11,'Actual133137 resource failure is not a normal0 predecessor')")
d=replace(d,'ASR18_readonly_saved3060_prefix_restore_with_retained_positive_tail_v1','ASR18_readonly_saved133000_prefix_restore_with_retained137_positive_tail_v1')
d=replace(d,'from fastmss_resident_feature_stream_v5 import ResidentFeatureProducer','from fastmss_resident_feature_stream_v6 import ResidentFeatureProducer')
driver_changes=copy.deepcopy(changes);changes=[]

old_wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v12.py';new_wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v15.py'
w=old_wrapper.read_text(encoding='utf-8');wrapper_old=w
w=replace(w,'run_ASR18_BPE9223_resident_RAM_training_v12','run_ASR18_BPE9223_resident_RAM_training_v15',2)
w=replace(w,'fastmss_resident_feature_stream_v5.py','fastmss_resident_feature_stream_v6.py')
w=replace(w,"        require(a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None,'New normal-boundary entry", "        require(a.abnormal_resume is None and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None,'New normal-boundary entry")
w=replace(w,"    elif a.original_interval_resume is not None:\n", "    elif a.abnormal_resume is not None:\n        require(a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None\n            and review['headroom_resume']==record(a.abnormal_resume)\n            and review['resource_scheduling_policy']==configuration['resident_RAM_resource_scheduling_policy']\n            and a.start_invocation==review['initial_owner_invocation']==11,\n            'Exact reviewed changed-workload abnormal133137 saved133000 entry, never normal0')\n    elif a.original_interval_resume is not None:\n")
w=replace(w,"    if a.original_interval_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v15 import checked_original_interval_resume", "    if a.abnormal_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v15 import checked_headroom_resume\n        checked_headroom_resume(record(a.abnormal_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    if a.original_interval_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v15 import checked_original_interval_resume")
w=replace(w,"            if a.original_interval_resume is not None:command+=['--original-interval-resume',a.original_interval_resume]", "            if a.abnormal_resume is not None:command+=['--abnormal-resume',a.abnormal_resume]\n            elif a.original_interval_resume is not None:command+=['--original-interval-resume',a.original_interval_resume]")
w=replace(w,"p.add_argument('--original-interval-resume');", "p.add_argument('--original-interval-resume');p.add_argument('--abnormal-resume');")
wrapper_changes=copy.deepcopy(changes)
cohort=[(old_cpu,new_cpu,cpu_old,c,cpu_changes),(old_driver,new_driver,driver_old,d,driver_changes),(old_wrapper,new_wrapper,wrapper_old,w,wrapper_changes)]
require(not D.exists() and all(not x[1].exists()for x in cohort),'Fresh inactive source namespace')
for old,new,before,after,items in cohort:
    ast.parse(after);inverse=after
    for item in reversed(items):
        require(inverse.count(item['new'])==item['count'],'Exact inverse replacement occurrence');inverse=inverse.replace(item['new'],item['old'])
    require(inverse==before,'Entire source equals original after only declared scheduling/typed-entry inverse')
D.mkdir()
for old,new,before,after,items in cohort:
    new.write_text(after,encoding='utf-8');(D/(new.stem+'.diff.txt')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old.name,tofile=new.name)),encoding='utf-8')
profile={'kind':'ASR18_resident_RAM_single_live_packet_resource_scheduling_v1','CPU_protocol_source':pin(new_cpu),
    'maximum_live_receiver_packets':1,'queue_capacity_bytes':268435456,'maximum_packet_bytes':268435456,
    'CPU_prospective_commit_reserve_bytes':2500000000,'CPU_allocation_wait_maximum_seconds':300,
    'CPU_actual_commit_headroom_bytes':1500000000,'native_actual_commit_headroom_bytes':2000000000,
    'native_actual_physical_headroom_bytes':1500000000,'CPU_private_limit_bytes':8000000000,
    'additional_prefetch_host_reservation_bytes':268435456,'freed_reservation_credit_claimed':False}
base_pin=N/'resident_RAM_original_interval_post88426_failure_actual_v1/production_configuration_v2/configuration.json'
require(pin(base_pin)['sha256']=='d48492be15373138dff2b1ff515255a2078f820cabf7c90e26de6370e88493c4','Actual unchanged source12 configuration')
base=read(base_pin);candidate=copy.deepcopy(base)
candidate['prior_resident_RAM_headroom_failure']=pin(N/'resident_RAM_headroom133137_failure_actual_v1/actual_failure_closure.json')
candidate['resident_RAM_resource_scheduling_policy']=profile
config_path=D/'configuration_candidate.json';write(config_path,candidate)
rest=copy.deepcopy(candidate);rest.pop('resident_RAM_resource_scheduling_policy');rest['prior_resident_RAM_headroom_failure']=base['prior_resident_RAM_headroom_failure'];require(rest==base,'Only two operational configuration fields')
handoff={'kind':'ASR18_INACTIVE_single_live_packet_resource_cohort_source_handoff_v1','production_admitted':False,
    'sources':{'CPU':pin(new_cpu),'driver':pin(new_driver),'wrapper':pin(new_wrapper)},'configuration':pin(config_path),
    'prior_configuration':pin(base_pin),'profile':profile,'actual_failure':candidate['prior_resident_RAM_headroom_failure'],
    'ROOT_failure_evidence_review':pin(N/'ROOT_headroom133137_failure_compact_and_ETA_review_v1.json'),
    'typed_initial_saved_updates':133000,'typed_initial_invocation':11,'future_new_journal':'commits_000011.jsonl',
    'historical_positive_tail_count':137,'normal_continuation_requires_actual_wait0_dead_all_live_saved':True,
    'source_maps':{new.name:{'old':pin(old),'new':pin(new),'exact_inverse_full_source':True,'declared_replacements':items,
        'diff':pin(D/(new.stem+'.diff.txt'))}for old,new,before,after,items in cohort},
    'numerical_prepare_native_commit_checkpoint_DEV_bodies_unchanged':True,'source14_timing_and_interval_preflight_inactive':True,
    'reservations_and_all_actual_guards_unchanged_or_stronger':True,'measured_saving_or_historical_queue_cause_claimed':False,
    'real_CPU_order_backpressure_exception_and_full_packet_control_pending':True}
write(D/'source_handoff_v1.json',handoff);print(json.dumps({'handoff':pin(D/'source_handoff_v1.json'),'sources':handoff['sources'],'configuration':pin(config_path)}),flush=True)
