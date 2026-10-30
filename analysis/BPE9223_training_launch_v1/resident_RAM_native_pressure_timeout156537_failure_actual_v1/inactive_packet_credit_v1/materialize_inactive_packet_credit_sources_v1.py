"""Metadata-only INACTIVE source derivation; never imports/executes candidate bodies."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).parent;N=F.parents[1]
def pin(p):
 b=Path(p).read_bytes();return {'path':str(Path(p)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,b):
 with Path(p).open('xb')as f:f.write(b)
def text(p):return Path(p).read_bytes().decode('utf-8').replace('\r\n','\n')
def changed(name,path,old,edits):
 new=old
 for before,after,count in edits:
  if new.count(before)!=count:raise ValueError((name,before,new.count(before),count))
  new=new.replace(before,after)
 ast.parse(new)
 inverse=new
 for before,after,count in reversed(edits):
  if inverse.count(after)!=count:raise ValueError(('inverse',name,after,inverse.count(after),count))
  inverse=inverse.replace(after,before)
 assert inverse==old
 write(path,new.replace('\n','\r\n').encode('utf-8'))
 write(F/(name+'.diff.txt'),''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=name+'_prior',tofile=str(path))).encode())
 changes[name]=[{'before':x,'after':y,'count':c}for x,y,c in edits]
 return new
changes={}
cpu_old=text(S/'fastmss_resident_feature_stream_v9.py')
policy_code='''\n\ndef checked_packet_credit_policy(config):
    from asr18_owned_packet_credit_v1 import source_pin
    policy=config['resident_RAM_packet_credit_policy']
    require(set(policy)=={'kind','helper_source','maximum_logical_packets','credit_wait_deadline_seconds',
        'original_packet_wait_seconds','actual_CPU_control_review'}
        and policy['kind']=='ASR18_owned_end_to_end_packet_credit_policy_v1'
        and policy['helper_source']==source_pin() and policy['maximum_logical_packets']==1
        and policy['credit_wait_deadline_seconds']==600 and policy['original_packet_wait_seconds']==300,
        'Exact separately reviewed named-credit transport policy; all old reservations unchanged')
    review=bound(policy['actual_CPU_control_review'])
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_owned_end_to_end_packet_credit_transport_v1'
        and review['helper_source']==policy['helper_source'] and review['CPU_source']==pin(__file__)
        and review['genuine_named_kernel_credit_and_pipe_flow_passed'] is True
        and review['synthetic_packets_no_audio_model'] is True,
        'Completed genuine kernel/transport CPU proof, honestly synthetic packet contents')
    return policy
'''
ready_code='''    def wait_update_ready(self, plan_sha, update_index, timeout=300):
        """Guarded NONCONSUMING wait: no pop/delivery/credit/authority change."""
        deadline=time.monotonic()+timeout
        with self.condition:
            while not self.packets:
                if self.error is not None:raise self.error
                if self.process.poll() is not None:
                    raise RuntimeError('CPU process ended without the requested packet; see CPU_stderr.log')
                if time.monotonic()>deadline:raise TimeoutError('Resident CPU update packet timed out')
                self.condition.wait(.25);self.guard()
            packet=self.packets[0]
            require(packet['plan_sha256']==plan_sha and packet['update_index']==update_index,
                    'Exact next packet ready without consuming or marking delivered')
            return {'plan_sha256':plan_sha,'update_index':update_index,'token':packet['token'],
                    'nonconsuming_readiness_only':True}

'''
cpu_edits=[
 ("ROOT = Path(r'E:\\gpu_training_scratchpad')","ROOT = Path(r'E:\\gpu_training_scratchpad')\nCREDIT_HOME = ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1'\nsys.path.insert(0,str(CREDIT_HOME))",1),
 ('\n\nclass AvailableMemoryPolicy:',policy_code+'\n\nclass AvailableMemoryPolicy:',1),
 ('    reader = None\n    completed = 0','    reader = None\n    packet_credit = None\n    completed = 0',1),
 ('    try:\n        core, single, source, validated, reader, augmenter, clean_reader, frontend = _startup',"    try:\n        from asr18_owned_packet_credit_v1 import PacketCreditConsumer\n        checked_packet_credit_policy(bound(configuration_path))\n        packet_credit=PacketCreditConsumer(json.loads(os.environ['ASR18_PACKET_CREDIT_DESCRIPTOR']))\n        core, single, source, validated, reader, augmenter, clean_reader, frontend = _startup",1),
 ("'startup_seconds': time.monotonic() - began})","'startup_seconds': time.monotonic() - began,'packet_credit':packet_credit.descriptor})",1),
 ('                update_began = time.monotonic()',"                packet_credit.acquire(phase='before_prepare_update',token={'plan_sha256':plan_sha,'update_index':index},check=policy)\n                update_began = time.monotonic()",1),
 ('        reader = None\n        _send(output,', '        reader = None\n        packet_credit.close()\n        _send(output,',1),
 ("'prospective_commit_reserve_bytes':policy.prospective_reserve_bytes})","'prospective_commit_reserve_bytes':policy.prospective_reserve_bytes,'packet_credit':packet_credit.receipt()})",1),
 ("'actual_resource_telemetry':{'last_observed_memory':policy.last_observed,","**({'packet_credit_timeout':exc.packet_credit_timeout}if hasattr(exc,'packet_credit_timeout')else{}),\n                'actual_resource_telemetry':{'last_observed_memory':policy.last_observed,",1),
 ('    finally:\n        if reader is not None:\n            reader.close()', '    finally:\n        try:\n            if reader is not None:reader.close()\n        finally:\n            if packet_credit is not None:packet_credit.close()',1),
 ('        profile=checked_resource_profile(config)\n        require(queue_bytes', '        profile=checked_resource_profile(config)\n        checked_packet_credit_policy(config)\n        require(queue_bytes',1),
 ("        self.log = (self.folder / 'CPU_stderr.log').open('xb')", "        from asr18_owned_packet_credit_v1 import PacketCreditOwner\n        self.credit=PacketCreditOwner()\n        environment['ASR18_PACKET_CREDIT_DESCRIPTOR']=json.dumps(self.credit.descriptor,sort_keys=True)\n        self.log = (self.folder / 'CPU_stderr.log').open('xb')",1),
 ('            self.error = exc\n            self.ready.set()', "            self.error = exc\n            try:self.credit.cancel()\n            except BaseException as cancel_error:self.credit_cancel_error=repr(cancel_error)\n            self.ready.set()",1),
 ("                    require(frame['source']['protocol'] == self.protocol_source, 'Unchanged CPU protocol source')", "                    require(frame['source']['protocol'] == self.protocol_source, 'Unchanged CPU protocol source')\n                    require(frame['packet_credit']==self.credit.descriptor,'Exact owned logical packet credit identity')",1),
 ("                    packet = {**frame, 'update': frame['metadata'], 'arrays': packed}\n                    with self.condition:", "                    packet = {**frame, 'update': frame['metadata'], 'arrays': packed}\n                    del arrays,packed,data,array,a,m # Packet alone owns payload before publication.\n                    with self.condition:",1),
 ('                        self.packets.append(packet)\n                        self.condition.notify_all()\n                    del arrays, packed, packet, data', '                        self.packets.append(packet)\n                        del packet # Drop receiver packet local before publication unlock/notification.\n                        self.condition.notify_all()',1),
 ("                    raise RuntimeError('Resident CPU preparation failed: ' + frame['message'])", "                    error=RuntimeError('Resident CPU preparation failed: ' + frame['message'])\n                    if 'packet_credit_timeout' in frame:error.packet_credit_timeout=frame['packet_credit_timeout']\n                    raise error",1),
 ('    def next_update(self, plan_sha, update_index, timeout=300):', ready_code+'    def next_update(self, plan_sha, update_index, timeout=300):',1),
 ("            value = self.live.pop(token)\n            require(value['delivered'], 'Cannot release an undelivered RAM packet')\n            self.used_bytes -= value['bytes']\n            self.condition.notify_all()", "            require(self.error is None,'Never release credit from errored transport')\n            value = self.live[token]\n            require(value['delivered'], 'Cannot release an undelivered RAM packet')\n            self.credit.return_credit() # Caller already dropped every payload reference.\n            self.live.pop(token)\n            self.used_bytes -= value['bytes']\n            self.condition.notify_all()",1),
 ("        result = {**self.closed_message, 'actual_Popen_wait_code': code,", "        require(set(self.plan_completions)==self.submitted,'Normal close only after every submitted plan completed')\n        require(self.closed_message['packet_credit']['all_owned_kernel_handles_closed'] is True,'Worker credit handles closed before normal report')\n        self.credit.close()\n        result = {**self.closed_message, 'packet_credit_owner':self.credit.receipt(), 'actual_Popen_wait_code': code,",1),
 ('        self.log.close()\n\n\nif __name__', '        self.log.close()\n        self.credit.close()\n\n\nif __name__',1)]
cpu_path=S/'fastmss_resident_feature_stream_v10.py';cpu=changed('CPU9_to10',cpu_path,cpu_old,cpu_edits)
helper_old=text(S/'asr18_resident_periodic_CPU_quiescence_v2.py')
helper=changed('quiescence2_to3',S/'asr18_resident_periodic_CPU_quiescence_v3.py',helper_old,[("            stream.release(packet['token'])\n            del packet, metadata","            token=packet['token']\n            del packet, metadata\n            stream.release(token)",1)])
validator='''def checked_packet_credit_resume(pin,*,base,folder,pointer_pin):
    """Actual added-buffer timeout, clean original monitor, saved156400+137tail."""
    import psutil
    require(pin==base['prior_resident_RAM_native_pressure156537_failure'] and record(pin['path'])==pin
        and pin['sha256']=='7e17d0ebf376135c554f2a86b98779376cdba1cc7756d4b585a4372806a05fb9',
        'Exact source22 TOOL1 pressure timeout; never original floor failure or normal0')
    value=read_json(pin['path']);e=value['inputs']
    require(value['kind']=='ASR18_actual_saved156400_source22_native_preupdate_pressure_timeout_closure_v1'
        and value['owned_session']==35083 and value['actual_outer_TOOL_exit_code']==1
        and value['actual_outer_TOOL_chunk_id']=='dd5bb7' and value['actual_native_optimizer_updates']==156537
        and value['last_fully_saved_optimizer_updates']==156400 and value['unsaved_positive_tail_count']==137
        and value['native_numeric_OS_exit_code'] is None and value['CPU_numeric_OS_exit_code'] is None
        and value['wrapper_numeric_OS_exit_code_separately_recorded'] is None
        and value['no_unknown_numeric_OS_exit_invented'] is True,
        'Actual unknown native/CPU/wrapper OS codes; real outer1 and137historical positives')
    for item in e.values():require(record(item['path'])==item,'Exact compact source22 timeout evidence before owner writes')
    require(e['pointer']==pointer_pin and record(pointer_pin['path'])==pointer_pin
        and e['driver']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v22.py')
        and e['CPU_protocol']==record(S/'fastmss_resident_feature_stream_v9.py')
        and e['effective_configuration']==record(folder/'invocations/owner_000017/configuration.json'),
        'Exact failed source22/CPU9 and genuine156400 checkpoint provenance')
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path'])
    d=failure['native_pressure_denial'];m=d['last_denied_sample']['system_memory']
    require(failure==value['actual_failure'] and failure['actual_optimizer_updates']==156537
        and failure['last_checkpoint']==156400 and failure['error']=="TimeoutError('Bounded native pre-update commit-pressure wait exhausted; preserve state, no unchanged retry')"
        and d['observed_wait_seconds']==300.0 and d['observations']==600
        and d['required_commit_available_bytes']==4768435456 and d['required_physical_available_bytes']==1768435456
        and m['commit_available_bytes']==4676124672 and m['physical_available_bytes']==11188850688
        and d['no_packet_propose_native_or_authoritative_operation_admitted'] is True
        and monitor==value['monitor_completion'] and monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,
        'Added-buffer scheduling timeout is distinct from original monitor/OOM/data failure')
    expected=[('native',12328,1792711453.1490774),('native_interpreter_launcher',38036,1792711453.1400745),
        ('wrapper',5628,1792711452.6819165),('CPU',28556,1792711652.8914447),('CPU_interpreter_launcher',37316,1792711652.8715985)]
    require([(x['role'],x['pid'],x['recorded_create_time'])for x in value['all_recorded_identities_dead']]==expected,'Exact five recorded predecessor identities')
    for role,pid,ctime in expected:
        try:now=psutil.Process(pid).create_time();dead=abs(now-ctime)>=.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Exact predecessor actually dead '+role)
    saved=value['saved_journal_prefix_as_verified_by_published_persistence'];stat=value['closed_journal_stat'];tail=value['preserved_unsaved_positive_tail']
    require(saved=={'path':str(folder/'journals/commits_000017.jsonl'),'offset':316802360,
        'sha256':'a5db7e40e8372c32c73b09b15ce87af7d3d9fed12bd8b1fede30f5fc3b461887'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved and stat['path']==saved['path']
        and stat['bytes']==317071928 and tail['bytes']==269568
        and tail['sha256']=='e5bf113be5ea2947b3fbbcc29d9467804dadb9742b941ca8b2c50d1b754b1ec3'
        and record(tail['path'])==tail and saved['offset']+tail['bytes']==stat['bytes'],
        'Exact published savedprefix and retained137tail; no invented whole17hash')
    current=Path(stat['path']).stat()
    require(current.st_size==stat['bytes'] and current.st_mtime_ns==stat['mtime_ns'],'Closed original17 STAT unchanged')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==137 and all(e['optimizer_updates']==e['native']['update']==e['prepared_evidence']['actual_positive_native_update']==u
        for u,e in zip(range(156401,156538),events)),'137 ordered original positive attempts remain historical')
    persistence=read_json(e['persistence']['path'])
    require(persistence['passed'] is True and persistence['optimizer_updates']==156400
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and persistence['journal_prefix']==saved and persistence['saved_semantic_state_verified']['passed'] is True,
        'Full saved156400854/207/215 PRE-SAVED and semantics; fresh restore still required')
    return {**value,'saved_journal_prefix':saved,'whole_journal_including_preserved_tail':stat,
        'preserved_unsaved_tail':tail,'preserved_tail_positive_updates':list(range(156401,156538))}


def check_original_closed_journal_stat(expected):
    observed=Path(expected['path']).stat()
    require(observed.st_size==expected['bytes'] and observed.st_mtime_ns==expected['mtime_ns'],
        'Original closed journal17 unchanged; no wholehash claimed or tail replay')
    return True


'''
driver_old=text(R/'run_ASR18_BPE9223_resident_RAM_training_v22.py')
driver_edits=[
 ('def checked_current_native_headroom_resume(',validator+'def checked_current_native_headroom_resume(',1),
 ('fastmss_resident_feature_stream_v9','fastmss_resident_feature_stream_v10',2),
 ('asr18_resident_periodic_CPU_quiescence_v2','asr18_resident_periodic_CPU_quiescence_v3',2),
 ("    abnormal=(checked_current_native_headroom_resume(record(a.native_headroom_resume),base=base,folder=folder,pointer_pin=pointer_pin)","    require(sum(x is not None for x in (a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume))<=1,'Only one typed abnormal resource lane')\n    packet_credit_failure=None if a.packet_credit_resume is None else checked_packet_credit_resume(\n        record(a.packet_credit_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    abnormal=(checked_current_native_headroom_resume(record(a.native_headroom_resume),base=base,folder=folder,pointer_pin=pointer_pin)",1),
 ('    frontend_failure=None if a.DEV_frontend_resume', '    if packet_credit_failure is not None:abnormal=packet_credit_failure\n    frontend_failure=None if a.DEV_frontend_resume',1),
 ('a.invocation==(17 if a.native_headroom_resume is not None else 11)','a.invocation==(18 if a.packet_credit_resume is not None else(17 if a.native_headroom_resume is not None else 11))',1),
 ("require(saved==source_journal and record(saved['path'])==abnormal['whole_journal_including_preserved_tail'],'Exact preserved failed journal')", "require(saved==source_journal and (check_original_closed_journal_stat(abnormal['whole_journal_including_preserved_tail'])if a.packet_credit_resume is not None else record(saved['path'])==abnormal['whole_journal_including_preserved_tail']),'Exact preserved failed journal')",1),
 ("('ASR18_readonly_saved156200_prefix_restore_with_retained100_positive_tail_v1' if a.native_headroom_resume is not None else 'ASR18_readonly_saved133000_prefix_restore_with_retained137_positive_tail_v1')", "('ASR18_readonly_saved156400_prefix_restore_with_retained137_positive_tail_v1' if a.packet_credit_resume is not None else('ASR18_readonly_saved156200_prefix_restore_with_retained100_positive_tail_v1' if a.native_headroom_resume is not None else 'ASR18_readonly_saved133000_prefix_restore_with_retained137_positive_tail_v1'))",1),
 ('record(a.native_headroom_resume if a.native_headroom_resume is not None else a.abnormal_resume)', 'record(a.packet_credit_resume if a.packet_credit_resume is not None else(a.native_headroom_resume if a.native_headroom_resume is not None else a.abnormal_resume))',1),
 ("require(record(source_journal['path'])==preserved_tail['whole_journal_including_preserved_tail'],'Old tail byteexact after prefix copy')", "require(check_original_closed_journal_stat(preserved_tail['whole_journal_including_preserved_tail'])if a.packet_credit_resume is not None else record(source_journal['path'])==preserved_tail['whole_journal_including_preserved_tail'],'Old tail byteexact after prefix copy')",1),
 ('((a.native_headroom_resume if a.native_headroom_resume is not None else a.abnormal_resume) if abnormal is not None else a.source_transition_resume)', '((a.packet_credit_resume if a.packet_credit_resume is not None else(a.native_headroom_resume if a.native_headroom_resume is not None else a.abnormal_resume)) if abnormal is not None else a.source_transition_resume)',1),
 ("                guard();packet=stream.next_update(digest(plan),index);metadata=packet.get('metadata',packet['update'])", "                stream.wait_update_ready(digest(plan),index) # Nonconsuming; includes current CPU preparation.\n                pressure_after_readiness=wait_for_native_commit_pressure(observe=monitor._observe,check=guard,policy=native_pressure_policy['parameters'])\n                metadata_timing['native_commit_pressure_wait_seconds']+=pressure_after_readiness['wait_seconds']\n                metadata_timing['native_commit_pressure_waited_updates']+=int(pressure_after_readiness['observations']>1 and pressure_admission['observations']<=1)\n                guard();packet=stream.next_update(digest(plan),index);metadata=packet.get('metadata',packet['update'])",1),
 ("prepared_archive.append({'binding':binding,'metadata':metadata,'native':native});stream.release(packet['token']);del packet,arrays", "prepared_archive.append({'binding':binding,'metadata':metadata,'native':native});token=packet['token'];del packet,arrays;stream.release(token)",1),
 ("**({'native_pressure_denial':error.native_pressure_denial}if hasattr(error,'native_pressure_denial')else{})", "**({'native_pressure_denial':error.native_pressure_denial}if hasattr(error,'native_pressure_denial')else{}),\n            **({'packet_credit_timeout':error.packet_credit_timeout}if hasattr(error,'packet_credit_timeout')else{})",1),
 ("p.add_argument('--native-headroom-resume');", "p.add_argument('--packet-credit-resume');p.add_argument('--native-headroom-resume');",1)]
driver=changed('driver22_to23',R/'run_ASR18_BPE9223_resident_RAM_training_v23.py',driver_old,driver_edits)
wrapper_old=text(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v22.py')
wrapper_edits=[
 ('run_ASR18_BPE9223_resident_RAM_training_v22','run_ASR18_BPE9223_resident_RAM_training_v23',7),
 ('fastmss_resident_feature_stream_v9','fastmss_resident_feature_stream_v10',1),
 ('asr18_resident_periodic_CPU_quiescence_v2','asr18_resident_periodic_CPU_quiescence_v3',1),
 ("require(a.native_headroom_resume is None and a.abnormal_resume", "require(a.packet_credit_resume is None and a.native_headroom_resume is None and a.abnormal_resume",1),
 ('    elif a.native_headroom_resume is not None:', "    elif a.packet_credit_resume is not None:\n        require(a.native_headroom_resume is None and a.abnormal_resume is None and a.original_interval_resume is None\n            and a.source_transition_resume is None and a.DEV_frontend_resume is None and a.DEV_admission_resume is None\n            and a.periodic_quiescence_resume is None and review['packet_credit_resume']==record(a.packet_credit_resume)\n            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']\n            and review['native_update_commit_pressure_policy']==configuration['native_update_commit_pressure_policy']\n            and a.start_invocation==review['initial_owner_invocation']==18,\n            'Exact source22 added-buffer TOOL1 saved156400 and reviewed end-to-end credit, never normal0')\n    elif a.native_headroom_resume is not None:",1),
 ("    if a.native_headroom_resume is not None:\n        from run_ASR18", "    if a.packet_credit_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v23 import checked_packet_credit_resume\n        checked_packet_credit_resume(record(a.packet_credit_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    from fastmss_resident_feature_stream_v10 import checked_packet_credit_policy\n    checked_packet_credit_policy(configuration)\n    if a.native_headroom_resume is not None:\n        from run_ASR18",1),
 ("            if a.native_headroom_resume is not None:command+=", "            if a.packet_credit_resume is not None:command+=['--packet-credit-resume',a.packet_credit_resume]\n            elif a.native_headroom_resume is not None:command+=",1),
 ("p.add_argument('--native-headroom-resume');", "p.add_argument('--packet-credit-resume');p.add_argument('--native-headroom-resume');",1)]
wrapper=changed('wrapper22_to23',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v23.py',wrapper_old,wrapper_edits)
base_path=N/'resident_RAM_system_headroom156300_failure_actual_v1/production_configuration_v1/configuration.json'
base=json.loads(base_path.read_text());config=copy.deepcopy(base)
config['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=pin(cpu_path)
config['periodic_DEV_CPU_quiescence_policy']['source']=pin(S/'asr18_resident_periodic_CPU_quiescence_v3.py')
config['resident_RAM_packet_credit_policy']={'kind':'ASR18_owned_end_to_end_packet_credit_policy_v1','helper_source':pin(F/'asr18_owned_packet_credit_v1.py'),
 'maximum_logical_packets':1,'credit_wait_deadline_seconds':600,'original_packet_wait_seconds':300,'actual_CPU_control_review':None}
config['prior_resident_RAM_native_pressure156537_failure']=pin(F.parent/'actual_failure_closure.json')
write(F/'configuration_template_NOT_ADMITTED.json',(json.dumps(config,indent=2,ensure_ascii=False)+'\n').encode())
report={'kind':'ASR18_INACTIVE_packet_credit_exact_derivation_v1','production_admitted':False,'control_executed':False,
 'all_full_text_inverses_equal':True,'line_endings':'original CRLF retained for CPU/driver/wrapper; prior helper2 LF normalized to CRLF with semantic inverse',
 'changes':changes,'source_base':pin(base_path),'source_configuration_template':pin(F/'configuration_template_NOT_ADMITTED.json'),
 'changed_config_top_keys':[k for k in config if config.get(k)!=base.get(k)],
 'limitations':['Completed CPU gate is null pending actual finite control and genuine ROOT acceptance.','No fit/savings/speed/one physical copy claim.','No candidate module imported or executed.']}
write(F/'source_inverse_and_configuration_report.json',(json.dumps(report,indent=2)+'\n').encode())
print(json.dumps({'sources':{p.name:pin(p)for p in [F/'asr18_owned_packet_credit_v1.py',cpu_path,R/'run_ASR18_BPE9223_resident_RAM_training_v23.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v23.py',S/'asr18_resident_periodic_CPU_quiescence_v3.py']},'inverse':pin(F/'source_inverse_and_configuration_report.json')}))
