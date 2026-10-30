"""New pending80/codec successors; admitted sources remain immutable."""
import ast,difflib,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')

def replace(s,a,b,n=1):
 assert s.count(a)==n,(a[:100],s.count(a),n)
 return s.replace(a,b)
def emit(p,s):
 assert not p.exists();ast.parse(s);p.write_text(s,encoding='utf8',newline='\n')
 return pin(p)
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

old=S/'asr18_BPE9223_full_run_controller_v18.py';text=old.read_text(encoding='utf8')
newdict=(N/'witness80_dictionary_v2_actual_01/dictionary_v2.bin').read_bytes()
constant="WITNESS_ZDICT_V2="+repr(newdict)+'''\n
def witness_decoder(packed):
    require(type(packed) is bytes and len(packed)>=6,'Complete zlib dictionary header')
    cmf,flg=packed[0],packed[1]
    require((cmf&15)==8 and cmf>>4<=7 and (cmf*256+flg)%31==0 and flg&32,
        'Only source-fixed dictionary zlib witness stream')
    dictionary_id=int.from_bytes(packed[2:6],'big')
    dictionaries={zlib.adler32(WITNESS_ZDICT):WITNESS_ZDICT,zlib.adler32(WITNESS_ZDICT_V2):WITNESS_ZDICT_V2}
    require(len(dictionaries)==2 and dictionary_id in dictionaries,'Only explicit legacy or V2 generic-field DICTID')
    return zlib.decompressobj(zdict=dictionaries[dictionary_id])

'''
at=text.index('\n',text.index('WITNESS_ZDICT='))+1;text=text[:at]+constant+text[at:]
text=replace(text,'decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)','decoder=witness_decoder(packed)',2)
text=replace(text,'decoder=zlib.decompressobj(zdict=WITNESS_ZDICT);raw=decoder.decompress(packed,65537)','decoder=witness_decoder(packed);raw=decoder.decompress(packed,65537)',0) if False else text
# The other two readers acquire their payload in the following expression.
text=replace(text,"guard();size,raw_size=struct.unpack('<II',header[:8]);decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)\n            raw=decoder.decompress(stream.read(size),65537)",
 "guard();size,raw_size=struct.unpack('<II',header[:8]);packed=stream.read(size);decoder=witness_decoder(packed)\n            raw=decoder.decompress(packed,65537)")
text=replace(text,'        decoder=zlib.decompressobj(zdict=WITNESS_ZDICT)\n        decoded=decoder.decompress(actual[40:],65537)',
 '        decoder=witness_decoder(actual[40:])\n        decoded=decoder.decompress(actual[40:],65537)')
text=replace(text,'raw=canonical(event);encoder=zlib.compressobj(level=9,zdict=WITNESS_ZDICT)',
 'raw=canonical(event);encoder=zlib.compressobj(level=9,zdict=WITNESS_ZDICT_V2)')
text=text.replace('from asr18_BPE9223_durable_owner_adapter_v8 import','from asr18_BPE9223_durable_owner_adapter_v9 import')
text=replace(text,'from asr18_BPE9223_persistence_prefix_v11 import OwnerPrefixVerifier',
 'from asr18_BPE9223_persistence_prefix_v12 import OwnerPrefixVerifier')

functions='''
def pending80_owner_state(failure_pin,*,guard=lambda:None):
    """Protocol60 is validation history; saved80 is the sole scientific frontier."""
    failed=read_bound(failure_pin,guard)
    require(failed['kind']=='ASR18_actual_committed80_witness_frame_archive_failure_TOOL1_v1'
        and failed['preserved_production_updates']==80 and failed['actual_auxiliary_archive_returned'] is True
        and failed['storage_publication_completed'] is False,'Exact returned archive and unadvanced durable protocol')
    snapshot=deepcopy(failed['previous_request_owner_state_preserved'])
    for key in ('actual_owner_invocation_count','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence','last_window_index'):
        snapshot[key]=failed[key]
    require([snapshot[k] for k in ('actual_owner_invocation_count','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence','last_window_index')]
        ==[13,5,3,5,11,2],'Exact cumulative failed80 counters, no attempt reuse')
    require(progress(normalize_schedule_state(snapshot['sampler_state']))['optimizer_updates']==60
        and snapshot['durable_head']==failed['current_durable_head'],'Existing acknowledged protocol60 only')
    registry=read_bound(failed['pool_registry'],guard);pool=Path(failed['pool_registry']['path']).parent
    snapshot['worker_control_pool_state']=failed['pool_registry']
    snapshot['worker_control_pool_binding']={'kind':'ASR18_exact_worker_control_pool_snapshot_v1',
        'owner':record(pool/'owner.json'),'registry':failed['pool_registry'],'index':failed['pool_index'],
        'archive_prefix':failed['pool_archive_prefix'],'archive_bytes':registry['archive_bytes'],'archive_count':registry['archive_count'],
        'reservation_count':registry['reservation_count'],'live_folders':[x['folder']for x in registry['live']],
        'owner_folders':registry['owner_folders']}
    require(registry['revision']==21 and registry['reservation_count']==7 and registry['archive_count']==0,'Actual stopped pool')
    snapshot['pending80']={'capture':failed['current_checkpoint_capture'],'packet':failed['completed_packet'],
        'completion':failed['current_native_completion'],'response':read_bound(failed['actual_native_response'],guard),
        'release':failed['current_native_release_ack'],'cleanup0':failed['actual_current_cleanup0'],
        'cleanup_report':failed['actual_current_cleanup_report'],'archive':failed['auxiliary_archive'],
        'unretired_witness':failed['unretired_operation_witness'],'mutation':failed['unretired_operation_mutation']}
    snapshot['prepared_CPU']=snapshot['reserved_preparation']=None
    return snapshot


def checked_pending80_resume_request(config_pin,review,request_pin,*,guard):
    from asr18_BPE9223_persistence_prefix_v12 import checked_committed80_witness_failure
    c=read_bound(config_pin,guard);r=read_bound(request_pin,guard)
    comp=checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
    fp=c['prior_committed80_witness_frame_failure']
    require(fp==comp['prior_committed80_witness_frame_failure']==r['prior_committed80_witness_frame_failure']
        and r['kind']=='ASR18_exact_completed80_pending_publication_witness_frame_resume_request_v1'
        and r['operational_configuration']==config_pin and review['resource_resume_request']==request_pin
        and r['operational_compatibility']==c['operational_compatibility']
        and r['authoritative_current_optimizer_updates']==80 and r['prior_durable_protocol_optimizer_updates']==60
        and r['no_source_optimizer_RNG_sampler_or_history_reset'] is True,
        'Exact pending80 completion, never replay60 or normal failed-owner0')
    failed=checked_committed80_witness_failure(fp,identity=c['full_run_identity'],operational_compatibility=c['operational_compatibility'],guard=guard)
    for key in ('storage_state','durable_latest','current_registered_journal','pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses'):
        require(record(failed[key]['path'])==failed[key]==r['current_stopped_state_bindings'][key],'Exact stopped80 authority '+key)
    expected=pending80_owner_state(fp,guard=guard);snapshot=deepcopy(r['prior_durable_owner_state'])
    snapshot['sampler_state']=normalize_schedule_state(snapshot['sampler_state'])
    require(digest(snapshot)==digest(expected),'Exact priorprotocol60/counters and actual saved80 inputs')
    require(c['session']==read_bound(failed['configuration'],guard)['session']
        and c['process_readonly_metadata_authority']==comp['process_readonly_metadata_authority'],
        'Frozen CPU6/session6 and stable readonly authority')
    return snapshot

'''
text=replace(text,'\nclass FullOwner:',functions+'\nclass FullOwner:')
text=replace(text,"require(request.get('kind') in ('ASR18_exact_prepared60_UTF8_preGPU_failure_resume_request_v1',",
 "require(request.get('kind') in ('ASR18_exact_completed80_pending_publication_witness_frame_resume_request_v1','ASR18_exact_prepared60_UTF8_preGPU_failure_resume_request_v1',")
text=replace(text,"        self.merged_CPU_inflight=False\n", "        self.merged_CPU_inflight=False\n        self.pending80_initial=False;self.pending80_finalization=False;self.pending80_inputs=None\n")
text=replace(text,"        self.committed60_initial=request.get('kind')", "        self.pending80_initial=request.get('kind')=='ASR18_exact_completed80_pending_publication_witness_frame_resume_request_v1'\n        self.committed60_initial=request.get('kind')")
text=replace(text,'self.resume_snapshot=(checked_prepared60_UTF8_resume_request(self.config_pin,r,resume_request,guard=lambda:None)',
 'self.resume_snapshot=(checked_pending80_resume_request(self.config_pin,r,resume_request,guard=lambda:None)\n            if self.pending80_initial else checked_prepared60_UTF8_resume_request(self.config_pin,r,resume_request,guard=lambda:None)')
text=replace(text,'snapshot=(checked_prepared60_UTF8_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)',
 'snapshot=(checked_pending80_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)\n                if self.pending80_initial else checked_prepared60_UTF8_resume_request(self.config_pin,r,resume_request,guard=self.parent.check)')
text=replace(text,"            self.pending40_inputs=deepcopy(snapshot.get('pending40'))", "            self.pending40_inputs=deepcopy(snapshot.get('pending40'))\n            self.pending80_inputs=deepcopy(snapshot.get('pending80'))")
text=replace(text,"acknowledged_committed60_resume=c['prior_acknowledged_committed60_stop'] if self.ACKed60_initial else None)",
 "acknowledged_committed60_resume=c['prior_acknowledged_committed60_stop'] if self.ACKed60_initial else None,\n                committed80_witness_resume=c['prior_committed80_witness_frame_failure'] if self.pending80_initial else None)")
text=replace(text,"            if request['kind'] in ('ASR18_exact_prepared60_UTF8_preGPU_failure_resume_request_v1',", "            if request['kind'] in ('ASR18_exact_completed80_pending_publication_witness_frame_resume_request_v1','ASR18_exact_prepared60_UTF8_preGPU_failure_resume_request_v1',")
text=replace(text,"        files.extend(x['configuration'] for x in state['live'] if x['folder'] in folders)",
 "        if self.config.get('prior_committed80_witness_frame_failure') is not None:\n            fp=self.config['prior_committed80_witness_frame_failure'];failed80=read_bound(fp,self.parent.check)\n            folders.add(str(Path(failed80['actual_native_boot']['path']).parent))\n            files.extend((fp,failed80['configuration'],failed80['resume_request'],failed80['owner_guard']))\n        files.extend(x['configuration'] for x in state['live'] if x['folder'] in folders)")
text=replace(text,'    def verify_durable_resume(self,request):\n', '''    def verify_durable_resume(self,request):
        if self.pending80_initial:
            require(request['committed80_witness_resume']==self.config['prior_committed80_witness_frame_failure']
                and request['fresh_recovery_still_required'] is True and request['no_authoritative_advance'] is True
                and request['state']['lifecycle']==self.resume_snapshot['lifecycle_state']
                and request['state']['pending']['windows']==request['state']['pending']['publications']==0
                and request['state']['last_recovery']==self.current_ack,'Only historical protocol60 reopen; actual80 restore still required')
            return True
''')
text=replace(text,'    def _current_capture_writer_source(self):\n', '''    def _current_capture_writer_source(self):
        if self.config.get('prior_committed80_witness_frame_failure') is not None:
            failed=read_bound(self.config['prior_committed80_witness_frame_failure'],self.parent.check)
            if self.current_capture is not None and self.current_capture['pointer']==failed['checkpoint_pointer']:
                return failed['runtime_source']
''')
text=replace(text,'            if self.pending40_finalization:self._verify_pending40_quiescence()',
 '            if self.pending80_finalization:self._verify_pending80_quiescence()\n            if self.pending40_finalization:self._verify_pending40_quiescence()')
text=replace(text,"(self.current_recovery_report if self.pending40_finalization else self.current_ack)","(self.current_recovery_report if self.pending40_finalization or self.pending80_finalization else self.current_ack)")
text=replace(text,'        if self.pending40_finalization:self._verify_pending40_quiescence()\n        else:verify_resident_idle',
 '        if self.pending80_finalization:self._verify_pending80_quiescence()\n        elif self.pending40_finalization:self._verify_pending40_quiescence()\n        else:verify_resident_idle')
text=replace(text,"        require(not self.pending40_initial,'No training/source20 restore before exact pending40 completion')",
 "        require(not self.pending40_initial and not self.pending80_initial,'No training before exact pending publication completion')")
methods='''
    def _verify_pending80_quiescence(self):
        require(self.pending80_finalization and self.gpu is None and self.closed_native==self.closed_recovery,
            'Only genuine fresh80 closed-state completion authority, never aborted native5 exit0')
        from asr18_BPE9223_persistence_prefix_v12 import checked_committed80_witness_failure
        failure=checked_committed80_witness_failure(self.config['prior_committed80_witness_frame_failure'],identity=self.identity,
            operational_compatibility=self.config['operational_compatibility'],guard=self.parent.check,reader=lambda p:read_bound(p,self.parent.check))
        report=read_bound(self.current_recovery_report,self.parent.check)
        verify(read_bound(self.closed_recovery,self.parent.check),kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,
            report=self.current_recovery_report,guard=self.parent.check)
        require(report['kind']=='ASR18_full_run_actual_CURRENT80_checkpoint_recovered_after_witness_frame_stop_v1'
            and report['current_checkpoint']==self.current_capture and self.current_capture==failure['current_checkpoint_capture']
            and report['current_committed80_witness_failure']==self.config['prior_committed80_witness_frame_failure']
            and report['actual_native_updates_in_recovery_process']==report['actual_draws_consumed_in_recovery']==0,
            'Actual exact80 fresh restore before completing returned archive/publication')
        return True

    def _finish_pending80(self):
        require(self.pending80_initial and self.schedule.committed_updates()==60 and self.gpu is None,
            'Protocol60 validation only; restore actual80 first, no prepare or draw')
        data=self.pending80_inputs;self.before_window=deepcopy(self.schedule.state_dict())
        self.current_response=deepcopy(data['response']);self.current_completion=data['completion']
        self.current_capture=deepcopy(data['capture']);self.current_release=data['release'];self.current_idle=None
        self.current_packet=deepcopy(data['packet']);self.current_intent=self.current_packet['checkpoint_intent']
        self.current_workspace=self.current_packet['auxiliary_workspace'];self.current_original_plan=self.current_packet['coordinator_plan']
        self.current_CPU0=self.current_packet['CPU_task_completion'];self.current_CPU_report=self.current_packet['CPU_task_report']
        self.current_bridge=read_bound(self.current_CPU_report,self.parent.check)['bridge']
        self.current_cleanup0=data['cleanup0'];self.current_cleanup=data['cleanup_report']
        require(self.native_sequence==5 and self.recovery_sequence==3 and self.recovery_control_sequence==5
            and self.actual_owner_invocation_count==14,'Exact counters after failed owner13/native5')
        self.recovery_control_sequence+=1;self.recovery_sequence+=1
        controls=self.control/('recovery_'+str(self.recovery_control_sequence).zfill(6));controls.mkdir(exist_ok=False)
        failed=read_bound(self.config['prior_committed80_witness_frame_failure'],self.parent.check)
        priorworker=read_bound(failed['actual_worker_configuration'],self.parent.check)
        original_first=priorworker['first_feature_update'];first=read_bound(original_first,self.parent.check)
        first_pin=self._new(controls/'full_forecast_reference_update.json',first)
        worker={**self.config,'worker_mode':'resident_training','committed80_witness_startup_restore':True,
            'resource_journal_handoff':False,'cleanup_join_journal_handoff':False,'committed60_CPU_journal_handoff':False,
            'source_checkpoint_capture':self.current_capture,'restore_parent_pointer':self.current_capture['pointer'],
            'journal_namespace':str(self.root/'journals'),'initial_journal_path':str(self.initial_journal),
            'journal_resume_sequence':self.recovery_sequence,'first_feature_update':first_pin,
            'original_full_forecast_reference_update':original_first,'actual_supervisor_identity':self.producer}
        cp=self._new(controls/'worker_configuration.json',worker)
        self.gpu=self.GPUService.launch(cp['path'],self.folder/('recovery_'+str(self.recovery_sequence).zfill(6)),self.parent)
        self.parent.GPU=self.gpu;ready=self._wait_ready(self.gpu.folder/'worker_ready.json')
        require(ready['optimizer_updates']==80,'Actual current80 startup before any command')
        self.closed_recovery=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        self.current_recovery_report=read_bound(self.closed_recovery,self.parent.check)['report'];self.closed_native=self.closed_recovery
        self.pending80_finalization=True;self._verify_pending80_quiescence()
        # Archive returned in failed invocation; retire that EXACT original proof,
        # never regenerate it or invoke archive_auxiliary a second time.
        self._retire_witness(data['unretired_witness'],data['mutation'])
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        metadata=read_bound(pointer['checkpoint'],self.parent.check)
        sampler=normalize_schedule_state(decode_scalar_tree(select(metadata['tree'],'sampler')))
        self.schedule.load_state_dict(sampler)
        original_proof=json.loads(read_bound(self.config['prior_committed80_witness_frame_failure'],self.parent.check)['unretired_operation_witness']['path']) if False else None
        plan=read_bound(self.current_original_plan,self.parent.check);last=plan['updates'][-1]
        self.current_cadence={'kind':'ASR18_source_bound_actual_completed_window_cadence_v1','phase':plan['phase'],
            'committed_window_updates':len(plan['updates']),'exact_pass_tail':bool(last['TR_draws'][-1]['draw_index']+1==
                (last['TR_draws'][-1]['epoch']+1)*(sampler['broad_cursor']['population'] if plan['phase']==WEAK else sampler['cleaner_cursor']['population'])),
            'coordinator_plan':self.current_original_plan}
        expectation={'identity':pointer['identity'],'optimizer_updates':80,'sampler_state':sampler,'journal_prefix':pointer['journal_prefix'],
            'run_identity':self.identity,'publication_role':'completed_window_end','publication_cadence_evidence':self.current_cadence}
        with self.parent.metadata_phase('read_index'):
            proof=self._owner_proof();published=self.store.complete_publication(self.current_intent,self.current_capture['pointer'],expectation,proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_checkpoint_pair_publication_verified','archive':published['archive']})
            self.durable.record_pending_window(before_state=self.before_window,plan=self.current_original_plan,completion=self.current_completion)
        with self.resolve_member({'kind':'ASR18_indexed_auxiliary_original_member_v1','original':self.current_original_plan,
                'storage_owner':self.store.marker_pin,'auxiliary_id':self.current_intent['id']})as stream:
            raw=stream.read(self.current_original_plan['bytes']+1)
        require(len(raw)==self.current_original_plan['bytes'] and hashlib.sha256(raw).hexdigest()==self.current_original_plan['sha256']
            and record(self.current_original_plan['path'])==self.current_original_plan,'Exact archived outside plan before retirement')
        Path(self.current_original_plan['path']).unlink();del raw,plan,metadata,sampler
        evidence={'kind':'ASR18_full_run_committed80_witness_current_recovery_prefix_inputs_v1','owner':OWNER,'diagnostic_only':False,
            'run_identity_sha256':digest(self.identity),'pending_head':self.durable.head_pin,'current_checkpoint_capture':self.current_capture,
            'current_committed80_witness_failure':self.config['prior_committed80_witness_frame_failure'],
            'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report,
            'operational_compatibility':self.config['operational_compatibility']}
        ep=self._new(controls/'evidence.json',evidence)
        request=self.durable._request(operational_compatibility=self.config['operational_compatibility'],current_recovery_evidence=ep,
            coverage_output=str(self.root/'durable'/('coverage_'+str(self.recovery_sequence).zfill(6)+'.json')))
        with self.parent.metadata_phase('read_index'):
            coverage=create_coverage(request,resolve_member=self.resolve_member,guard=self.parent.check);self.current_coverage=coverage
            value=read_bound(coverage,self.parent.check)
            ack={'kind':'ASR18_committed80_witness_failure_current_recovery_and_verified_persistence_prefix_v1',
                'diagnostic_only':False,'run_identity_sha256':digest(self.identity),'pending':deepcopy(self.durable.head['pending']),
                'pending_head':self.durable.head_pin,'final_checkpoint_pointer':self.current_capture['pointer'],
                'fresh_recovery_scope':'current_latest_only','all_persistence_verified_through_pending_prefix':True,
                'all_historical_checkpoints_freshly_restored':False,
                'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,
                'actual_recovery_native_updates':0,'actual_recovery_draws_consumed':0,
                'actual_committed80_witness_quiescence_proven':True,'actual_native_normal_exit0_claimed':False,
                'current_committed80_witness_failure':self.config['prior_committed80_witness_frame_failure'],
                'actual_closed_native_index':self.closed_recovery,'actual_closed_recovery_index':self.closed_recovery,
                'current_closed_authority_role':'genuine_fresh_restored80_process0_never_original_training_exit0',
                'persistence_prefix_coverage_index':coverage,'current_latest_exact_restore':value['current_latest_exact_restore'],
                'current_recovery_evidence':ep,'operational_compatibility':self.config['operational_compatibility']}
            self.current_ack=self._new(controls/'acknowledgement.json',ack);self.durable.acknowledge_recovery(self.current_ack)
        self.last_window_index=3;self.invocation_initial_updates=80
        require(self.schedule.committed_updates()==80 and self.durable.head['pending']['publications']==0,'Published and ACKed80 before81')
        self.pending80_finalization=False;self.pending80_initial=False;self.pending80_inputs=None
        self._refresh_boundary();self._retire_obsolete();self._archive_obsolete_worker_controls();gc.collect()

'''
text=replace(text,'    def _finish_pending40(self):',methods+'    def _finish_pending40(self):')
text=replace(text,'    def run(self):\n        if self.committed60_initial:', '    def run(self):\n        if self.pending80_initial:self._finish_pending80()\n        if self.committed60_initial:')
controller=emit(S/'asr18_BPE9223_full_run_controller_v19.py',text)

oldruntime=R/'asr18_BPE9223_owned_GPU_runtime_v23.py';runtime=oldruntime.read_text(encoding='utf8')
helper='''
def checked_committed80_startup(config,*,guard=lambda:None):
    from asr18_BPE9223_persistence_prefix_v12 import checked_committed80_witness_failure
    from asr18_BPE9223_operational_metadata_v2 import read_bound
    failure=checked_committed80_witness_failure(config['prior_committed80_witness_frame_failure'],identity=config['full_run_identity'],
        operational_compatibility=config['operational_compatibility'],guard=guard,reader=lambda p:read_bound(p,guard))
    require(config['committed80_witness_startup_restore'] is True and config['worker_mode']=='resident_training'
        and config['source_checkpoint_capture']==failure['current_checkpoint_capture']
        and config['restore_parent_pointer']==failure['checkpoint_pointer']
        and config['parent_driver_source']==failure['source']
        and config['entire_worker_forecast']['entire_worker_private_forecast_bytes']==7997712768,
        'Exact saved80 fresh zero-update startup under whole native forecast and frozen19 services')
    return failure

'''
runtime=replace(runtime,'\ndef run(config_path,folder):',helper+'\ndef run(config_path,folder):')
runtime=replace(runtime,'    def journal_factory(saved):\n', '''    def journal_factory(saved):
        if config.get('committed80_witness_startup_restore') is True:
            stopped=checked_committed80_startup(config,guard=guard)
            expected={'path':stopped['current_registered_journal']['path'],'offset':stopped['current_registered_journal']['bytes'],
                'sha256':stopped['current_registered_journal']['sha256']}
            require(saved==expected and Path(saved['path'])in namespace.paths,'Exact whole80 journal, no suffix replay/rollback')
            namespace.check();journal=namespace.writer(Path(saved['path']));require(journal.binding()==saved,'Actual unchanged80 journal')
            return journal,{'kind':'ASR18_committed80_witness_closed_journal_resume_v1','source':record(__file__),
                'current_committed80_witness_failure':config['prior_committed80_witness_frame_failure'],
                'source_checkpoint_prefix':saved,'journal':journal.binding(),'actual_native_normal_exit0_claimed':False,
                'copied_bytes':0,'tail_replayed_or_sampler_advanced':False}
''')
runtime=replace(runtime,"            engine.restore_parent(config['restore_parent_pointer'],augmentation_identity=config['augmentation_identity'])", '''            engine.restore_parent(config['restore_parent_pointer'],augmentation_identity=config['augmentation_identity'])
            if config.get('committed80_witness_startup_restore') is True:
                failure80=checked_committed80_startup(config,guard=guard)
                engine.capture=copy.deepcopy(config['source_checkpoint_capture'])
                engine.last_window=failure80['current_native_completion']
                from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
                from asr18_checkpoint_fingerprint_normalizer_v1 import normalize_fingerprint
                from asr18_BPE9223_operational_metadata_v2 import read_bound
                actual80=live_state_fingerprint(optimizer,torch=torch,guard=guard)
                expected80=normalize_fingerprint(read_bound(engine.capture['pre_save_fingerprint'],guard))
                require_exact_live_restore(expected80,actual80)
                restored80=read_json(folder/'actual_parent_restore.json')
                require(optimizer.updates==engine.process_initial_updates==80 and restored80['passed'] is True
                    and digest(engine.schedule.state_dict())==read_bound(engine.capture['pointer'],guard)['sampler_state_sha256'],
                    'Actual all854/207/threeAdam/allRNG/samplers/nextdraw80 restored with zero updates')
                core80={'passed':True,'pointer':engine.capture['pointer'],'checkpoint':engine.capture['checkpoint'],
                    'pre_save_fingerprint':engine.capture['pre_save_fingerprint'],'next_draw_packet':engine.capture['next_draw_packet'],
                    'actual_restored_state_sha256':actual80['state_sha256'],'updates':80,
                    'all207_threeAdam_all4RNG_match_actual_pre_save':True,'both_samplers_journal_prefix_and_next_draw_exact':True,
                    'branch':restored80['branch'],'native_updates_performed':0,'draws_consumed':0,'separate_fresh_GPU_recovery':True}
                write_json(folder/'actual_current80_recovery.json',core80)
''')
runtime=replace(runtime,"            require(engine is not None,'Recovery cannot consume native commands')", "            require(engine is not None and config.get('committed80_witness_startup_restore') is not True,'Recovery cannot consume native commands')")
runtime=replace(runtime,"    del engine,optimizer,teacher,model,processor;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()", '''    if config.get('committed80_witness_startup_restore') is True:
        require(result['actual_native_updates_this_process']==0 and result['final_optimizer_updates']==80,'No update/draw in separate actual80 recovery')
        result.update(kind='ASR18_full_run_actual_CURRENT80_checkpoint_recovered_after_witness_frame_stop_v1',source=record(__file__),
            current_committed80_witness_failure=config['prior_committed80_witness_frame_failure'],
            operational_compatibility=config['operational_compatibility'],recovered_checkpoints=[record(folder/'actual_current80_recovery.json')],
            current_latest_checkpoint_recovered=True,historical_fresh_GPU_restores_claimed=False,
            actual_native_updates_in_recovery_process=0,actual_draws_consumed_in_recovery=0)
    del engine,optimizer,teacher,model,processor;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()''')
runtimepin=emit(R/'asr18_BPE9223_owned_GPU_runtime_v24.py',runtime)
area=N/'pending80_witness_codec_source_v1';area.mkdir(exist_ok=False)
for label,a,b in [('controller18_to19',old,S/'asr18_BPE9223_full_run_controller_v19.py'),('runtime23_to24',oldruntime,R/'asr18_BPE9223_owned_GPU_runtime_v24.py')]:
 (area/(label+'.diff.txt')).write_text(''.join(difflib.unified_diff(a.read_text(encoding='utf8').splitlines(True),b.read_text(encoding='utf8').splitlines(True),fromfile=str(a),tofile=str(b))),encoding='utf8')
(area/'source_handoff_draft.json').write_text(json.dumps({'controller':controller,'runtime':runtimepin,'native_unchanged':pin(S/'asr18_BPE9223_full_run_native_v9.py'),
    'parent_unchanged':pin(R/'run_ASR18_BPE9223_full_run_owner_v19.py')},indent=2)+'\n',encoding='utf8')
print(json.dumps({'controller':controller,'runtime':runtimepin,'folder':str(area)}))
