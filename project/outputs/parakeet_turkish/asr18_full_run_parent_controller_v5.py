"""Concrete inactive full-run owner body; bounded validation uses its stop budget.

Normal windows remain resident and publish persistence once at window-end.
Actual native exit and CURRENT fresh recovery precede durable prefix ack, DEV,
winner selection, and phase advance. Validation never fabricates a pass barrier.
"""
from copy import deepcopy
from contextlib import contextmanager
import gc,hashlib,inspect,json,os,secrets,shutil,time,struct,zlib
from pathlib import Path
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import canonical,digest
from bpe_full_run_lifecycle_v2 import FullRunLifecycle,SourceBoundOwnerCallbacks,CALLBACKS,REFERENCES,LIMITS,progress
from bpe_full_run_durable_owner_adapter_v1 import OwnerProofCallbacks
from bpe_full_run_durable_owner_adapter_v2 import PackedDurableLifecycleAdapter
from bpe_indexed_owned_checkpoint_storage_v3 import IndexedOwnedFutureCheckpointStore,AUX_ROLES
from bpe_bounded_production_journal_v4 import JournalNamespace
from asr18_full_run_source_context_v1 import FullRunSourceContext
from asr18_full_run_persistence_prefix_v1 import OwnerPrefixVerifier,create_coverage,_Metadata,_capture,_JournalPrefixes
from asr18_full_run_metadata_resolver_v1 import MetadataResolver
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from bpe_training_schedule_v1 import WEAK
from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree,verify_saved_metadata
from fastmss_phase_aware_plan_bridge_v7 import bridge_plan,read_bound
from fastmss_owned_process_closure_v2 import identity,verify,verify_resident_idle
ROOT=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis'


class FullOwner:
    def __init__(self,config_path,review_path,folder,*,parent_source,ParentGuard,GPUService,run_CPU,cleanup_script,
                 storage,fresh_admit,execute_owned_DEV,native_boundary_DEV,install_owner):
        storage();self.config_pin=record(config_path);self.config=read_json(config_path);self.review_pin=record(review_path)
        self.review=read_json(review_path);self.parent_source=parent_source;self.GPUService=GPUService;self.run_CPU=run_CPU
        self.cleanup_script=cleanup_script;self.fresh_admit=fresh_admit;self.producer=identity();self.gpu=None
        self.folder=Path(folder).resolve();self.root=Path(self.config['run_namespace']).resolve();self.control=self.root/'control'
        self.identity=deepcopy(self.config['full_run_identity']);self.closed_native=None;self.closed_recovery=None
        self.current_capture=None;self.current_completion=None;self.current_release=None;self.current_idle=None
        self.current_recovery_report=None;self.current_ack=None;self.current_coverage=None;self.prior_boundary=None
        self.current_packet=None;self.last_window_index=-1;self.resume_checked=False;self.DEV=None
        self.native_sequence=0;self.recovery_sequence=0;self.proof_sequence=0;self.boundary_state=None
        c=self.config;r=self.review
        require(c['kind']=='ASR18_full_run_native_owner_configuration_v1' and c['production_enabled'] is False
                and c['diagnostic_only'] is False and c['planned_native_updates']==401072
                and c['parent_driver_source']==parent_source and record(parent_source['path'])==parent_source,
                'Exact original full-run owner and immutable source configuration')
        require(r['accepted'] is True and r['reviewer']=='/root' and r['configuration']==self.config_pin
                and r['source_manifest']==c['source_manifest'] and r['production_approval_claimed'] is False,
                'Exact separate source/resource/execution review')
        if c['owner_validation_only']:
            require(r['scope']=='finite_full_run_owner_four_update_pipeline_validation_v1'
                    and c['maximum_owner_validation_updates']==c['maximum_validation_updates_per_window']==4
                    and self.identity['owner_validation_only'] is True
                    and self.identity['production_source_eligible'] is False,
                    'ONE exact four-update owner validation, no diagnostic seed promotion')
        else:
            require(c['full_run_launch_enabled'] is True and r['full_run_launch_enabled'] is True
                    and r['scope']=='reviewed_full_two_broad_one_cleaner_actual_launch_v1'
                    and self.identity['owner_validation_only'] is False,
                    'Long launch requires separately exact human-authorized configuration; no CLI override')
        require(self.folder.is_relative_to(ROOT) and self.root.is_relative_to(ROOT/'runs')
                and not self.folder.exists() and not self.root.exists(),'Fresh exact owner namespaces, never existing artifact adoption')
        self.folder.mkdir();self.root.mkdir();self.control.mkdir();self.parent=ParentGuard(self.folder)
        self.parent.configure_control_quota(self.control,c['control_log_policy'])
        self.began=time.monotonic();self.current_worker_config=None
        self.context=FullRunSourceContext(c['planning_arguments'],guard=self.parent.check);self.schedule=self.context.new_schedule()
        require(digest(self.schedule.state_dict())==c['initial_sampler_sha256'] and self.schedule.committed_updates()==0,
                'Original source BPE64250/LoRA62250 RNG/sampler frontier0')
        for i in range(3):
            for extra in (c['entire_worker_forecast']['entire_worker_private_forecast_bytes']+384000000+64000000,
                          c['CPU_owned_allowance_bytes']+128000000):
                require(fresh_admit(extra)['passed'],'Three fresh whole-domain admissions before namespace/producer work')
            time.sleep(.2)
        self.journal_namespace=JournalNamespace(self.root/'journals',identity=self.identity,guard=self.parent.check,create=True)
        self.initial_journal=self.journal_namespace.register('commits_000000.jsonl')
        lifecycle_functions={name:getattr(self,name) for name in CALLBACKS}
        callbacks=SourceBoundOwnerCallbacks(sources={name:record(__file__) for name in CALLBACKS},callbacks=lifecycle_functions)
        self.lifecycle=FullRunLifecycle(self.schedule,identity=self.identity,callbacks=callbacks,
            qualified_winner=deepcopy(c['qualified_source_winner']))
        prefix=OwnerPrefixVerifier(resolve_member=self.resolve_member,guard=self.parent.check)
        methods={'verify_pending_window':self.verify_pending_window,'verify_writer_authority':self.verify_writer_authority,
            'verify_durable_resume':self.verify_durable_resume,
            'verify_current_recovery_and_persistence_prefix':prefix.verify_current_recovery_and_persistence_prefix}
        proofs=OwnerProofCallbacks(sources={name:record(inspect.getsourcefile(function)) for name,function in methods.items()},functions=methods)
        with self.parent.metadata_phase('read_index'):
            self.durable=PackedDurableLifecycleAdapter(self.lifecycle,self.root/'durable',policy=c['durable_policy'],proofs=proofs,create=True)
            require(self.durable.owner==read_bound(c['session'],self.parent.check)['durable_owner'],
                    'Actual newly written deterministic owner equals session expected binding, never an invented actual receipt')
            self.store=IndexedOwnedFutureCheckpointStore.create(c['checkpoint_storage_namespace'],c['checkpoint_storage_policy'],
                ROOT_review=c['ROOT_storage_review'],run_identity=self.identity,verifier_source=record(__file__),
                protected=c['protected_external'],guard=self.parent.check)
        self.resolver=MetadataResolver(self.store,control_root=self.control,guard=self.parent.check,metadata_phase=self.parent.metadata_phase)
        # Prefix's resolver is this exact source-bound wrapper, delegated to the
        # separately frozen concrete resolver; identity pins this controller too.
        self.prefix=prefix;self.proofs=proofs;install_owner(self)
        from asr18_full_run_serial_DEV_v9 import SerialDEVCallbacks
        self.DEV=SerialDEVCallbacks(ROOT_policy_review=c['ROOT_DEV_policy_review'],execution_sources=c['DEV_execution_sources'],
            execute_owned=execute_owned_DEV,execute_owned_source=parent_source,native_boundary=native_boundary_DEV,
            native_boundary_source=parent_source,native_boundary_sources=c['DEV_native_boundary_sources'],output_directory=self._DEV_folder())
        write_json(self.folder/'launch.json',{'configuration':self.config_pin,'review':self.review_pin,'producer':self.producer,
            'run_identity_sha256':digest(self.identity),'production_enabled':False})

    def _new(self,path,value):
        path=Path(path).resolve()
        require((path.is_relative_to(self.root) or path.is_relative_to(self.folder)) and not path.exists(),
                'Fresh immutable exact owned control artifact')
        if path.is_relative_to(self.control):self.parent.control_guard(path,len(canonical(value))+4096)
        write_json(path,value);return record(path)

    def _retire_witness(self,pin,mutation):
        # An immutable range in ONE bounded source-owned control stream keeps
        # every original witness byte/pin, including retirement operations whose
        # store event itself only contains the original witness binding.
        require(record(pin['path'])==pin,'Same returned-operation witness before exact owned retirement')
        path=Path(pin['path']).resolve()
        require(path.parent==self.control and '.proof.' in path.name,'Only exact new operation witness')
        original=path.read_bytes()
        event={'kind':'ASR18_exact_archived_operation_witness_v1','original':pin,
            'mutation':mutation,'raw_UTF8':original.decode('utf-8')}
        raw=canonical(event);packed=zlib.compress(raw,9)
        require(len(raw)<=65536 and len(packed)<=2048,'Complete lossless witness frame cap; no field dropping')
        log=self.control/'storage_operation_witnesses.bin'
        offset=log.stat().st_size if log.exists() else 0
        frame=struct.pack('<II',len(packed),len(raw))+hashlib.sha256(raw).digest()+packed
        require(offset+len(frame)<=805306368,'Entire declared operation-witness control horizon')
        self.parent.control_guard(log,offset+len(frame))
        with log.open('ab') as stream:stream.write(frame);stream.flush();os.fsync(stream.fileno())
        with log.open('rb') as stream:stream.seek(offset);actual=stream.read(len(frame))
        decoded=zlib.decompress(actual[40:])
        require(actual==frame and decoded==raw and json.loads(decoded)['raw_UTF8'].encode('utf-8')==original
            and hashlib.sha256(original).hexdigest()==pin['sha256'] and len(original)==pin['bytes'],
            'Actual fsynced immutable complete witness range/readback before single owned unlink')
        path.unlink();require(not path.exists(),'Actual exact transient witness retirement')

    def _DEV_folder(self):
        path=self.root/'DEV';path.mkdir(exist_ok=False);return path

    @contextmanager
    def resolve_member(self,binding):
        require(hasattr(self,'resolver'),'Archive resolver unavailable before namespace initialization')
        with self.resolver(binding) as stream:yield stream

    def check(self):
        self.parent.check();require(identity()==self.producer and record(self.config_pin['path'])==self.config_pin
            and record(self.parent_source['path'])==self.parent_source,'Actual same source-bound owner process/configuration')
        lease=read_json(Path(self.parent_source['path']).resolve().parents[2]/'work/gpu_training_owner.json')
        require(lease['owner']==OWNER,'Exclusive native owner unchanged')
        if self.gpu is None:require(not lease['actual_GPU_running'] and not lease.get('actual_owned_worker_pid'),'No unknown live GPU child')
        else:require(lease['actual_GPU_running'] and lease['actual_owned_worker_pid']==self.gpu.pid,'Exact actual owned GPU child')
        return True

    def verify_writer_authority(self,request):
        self.check();require(request['identity']==self.identity and request['namespace']==str(self.root/'durable')
            and request['policy']==self.config['durable_policy'],'Exact sole durable writer authority')
        return True

    def verify_pending_window(self,request):
        self.check();require(self.gpu is not None and self.current_idle is not None and request['identity']==self.identity
            and request['plan']==self.current_packet['coordinator_plan'] and request['completion']==self.current_completion,
            'Actual current owned window, not report existence alone')
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,phase=self.current_packet['phase'],
            proof_binding=self.current_idle,guard=self.parent.check)
        io=_Metadata(self.identity,self.resolve_member,self.parent.check);journal=_JournalPrefixes(self.parent.check)
        try:
            _,after=_capture(self.current_capture,self.current_capture['pointer'],io,identity_sha=digest(self.identity),
                previous_sampler=digest(request['before_state']),previous_updates=progress(request['before_state'])['optimizer_updates'],journal=journal)
            require(digest(after)==digest(request['expected_after_state'])==digest(self.schedule.state_dict())
                    and request['all_captures']==[self.current_capture],'Exact PRE/persistence current actual sampler prefix')
        finally:journal.close()
        return True

    def verify_durable_resume(self,request):
        self.check();require(self.gpu is None and self.current_ack is not None and request['restore_proof']==self.current_ack
            and request['identity']==self.identity and request['namespace']==str(self.root/'durable')
            and request.get('native_replay_or_sampler_rollback_permitted') is False,'Exact current recovery before same-owner durable reopen')
        value=read_bound(self.current_ack,self.parent.check)
        require(request['state']['last_recovery']==self.current_ack and request['state']['pending']['publications']==0
                and value['final_checkpoint_pointer']==self.current_capture['pointer'],'No older lifecycle restore or unreconciled orphan')
        self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
        return True

    def verify_resume(self,request):
        require(self.current_ack is not None and request['proof']==self.current_ack and request['identity']==self.identity,
            'Actual exact current recovery before lifecycle reconstruction')
        value=read_bound(self.current_ack,self.parent.check)
        require(request['lifecycle']['last_pointer']==value['final_checkpoint_pointer']
                and digest(request['schedule'])==self.lifecycle.state_dict()['last_sampler_sha256'],'Current authoritative lifecycle/schedule')
        return True

    def verify_winner(self,request):
        winner=request['winner']
        if winner==self.config['qualified_source_winner']:
            summary=read_bound(self.config['qualified_source_DEV_summary'],self.parent.check)
            errors={lang:summary['validation_'+lang]['asr']['word_errors'] for lang in REFERENCES}
            require(winner['strict_qualified'] is True and winner['word_errors']==errors and winner['reference_words']==REFERENCES
                and all(summary['validation_'+l]['asr']['reference_words']==n for l,n in REFERENCES.items())
                and all(errors[l]<=cap for l,cap in LIMITS.items())
                and winner['artifact']==self.config['planning_arguments']['expected_original_sources']['parked_parent'],
                'Exact qualified original source/winner; no diagnostic promotion')
            return True
        artifact=winner.get('artifact') if type(winner) is dict and 'artifact' in winner else winner
        report=request.get('DEV',winner.get('DEV') if type(winner) is dict else None)
        require(report is not None and self.current_ack is not None,
                'Actual independently sealed qualified winner, never a diagnostic candidate')
        actual=read_bound(report,self.parent.check)
        source_request=read_bound(actual['source_request'],self.parent.check)
        from asr18_full_run_serial_DEV_v9 import verified_completed_DEV
        require(verified_completed_DEV(actual['source_request'],actual['actual_worker_report'],actual['actual_owned_DEV_process0'])==actual,
                'Same sealed actual official350 execution, never a retention fixture')
        from bpe_full_run_lifecycle_v2 import official_DEV
        score=official_DEV(actual,artifact,read_bound(self.identity['scorer_policy'],self.parent.check))
        require(score['strict_all_six_passed'] and source_request['identity']==self.identity
                and source_request['checkpoint_pointer']==artifact,
                'Exact integer all-six retention and original immutable candidate/run identity')
        if type(winner) is dict and 'artifact' in winner:
            with self.parent.metadata_phase('read_index'):
                with self.store.locked():
                    state=self.store._load()[0]
                    entries=[self.store._entry(key,verify_files=True) for key in state['winners']]
                require(any(entry['pointer']==artifact and entry.get('qualification') for entry in entries),
                        'Historical qualified winner remains actually registered/protected with its original immutable DEV')
        else:
            require(artifact==self.current_capture['pointer'],'Only current recovered candidate can be NEWLY promoted')
        if type(winner) is dict and 'artifact' in winner:
            require(winner['strict_qualified'] and winner['word_errors']==score['word_errors']
                and winner['reference_words']==REFERENCES,'Stored winner exact official score')
        return True

    def admit_operation(self,request):self.check();return True
    def prepare_window(self,request):
        raise ValueError('Owner drives lifecycle.request + pass-bounded plan directly; lifecycle.advance is intentionally unused')
    def execute_window(self,request):
        raise ValueError('Owner drives actual resident persistence + durable pending callback, never fake per-window restore')
    def verify_checkpoint(self,request):
        require(request['pointer']==self.current_capture['pointer'] and request['completion']==self.current_completion,'Current actual checkpoint');return True
    def checkpoint_storage_boundary(self,request):return self.config['ROOT_storage_boundary_receipt']
    def serial_DEV(self,request):return self.DEV.serial_DEV(request)
    def verify_DEV(self,request):return self.DEV.verify_DEV(request)
    def preserve_winner(self,request):
        if not request['replace']:return request['old_winner']['artifact']
        require(request['score']['strict_all_six_passed'] and self.gpu is None and self.current_ack is not None,
                'Actual current fresh recovery and official strict qualification before winner protection')
        self.verify_winner({'winner':request['candidate_pointer'],'DEV':request['DEV']})
        with self.parent.metadata_phase('read_index'):
            proof=self._owner_proof(extra={'qualification':request['DEV'],'strict_all_six_qualified':True,
                'qualified_pointer':self.current_capture['pointer']})
            self.store.protect_qualified_winner(self.current_intent['id'],proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_qualified_winner_protected','checkpoint_id':self.current_intent['id']})
        return request['candidate_pointer']

    def _verify_closed(self,pin):
        proof=read_bound(pin,self.parent.check);verify(proof,kind=proof['kind'],owner=OWNER,guard=self.parent.check);return proof

    def _read_state(self):
        with self.store.locked():return self.store._load(allow_preparing_auxiliary=True)[0]

    def _owner_proof(self,*,extra=None):
        self.check();require(self.current_release is not None,'Actual returned checkpoint writer/read/restore contexts')
        state=self._read_state();release=read_bound(self.current_release,self.parent.check)
        require(release['worker_source']==self.config['worker_driver_source']
                and release['request_nonce']==self.current_response['nonce'] and release['writer_intent']==self.current_intent
                and release['checkpoint_pointer']==self.current_capture['pointer']
                and all(release[k] is True for k in ('writer_context_returned','persistence_reader_context_returned','startup_restore_context_returned')),
                'Exact candidate checkpoint contexts returned before publication/archive')
        if self.gpu is not None:
            self.gpu.check();require(release['producer']==self.gpu.producer,'Same actual live native writer')
        else:
            require(self.current_ack is not None,'Closed-state mutation requires current exact recovery')
            self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
        latest=None if state['latest'] is None else self.store._entry(state['latest'],verify_files=False)['pointer']
        value={'kind':'ASR18_owned_checkpoint_mapping_release_quiescence_v1','owner':OWNER,'ownership':self.store.marker_pin,
            'state_revision':state['revision'],'catalog':state['catalog'],'coherent_latest_pointer':latest,
            'protected_owned_winners':state['winners'],'nonce':secrets.token_hex(32),
            'native_work_quiescent':True,'checkpoint_mappings_released':True,'candidate_payload_handles_released':True,
            'checkpoint_write_inflight':False,'restore_in_progress':False,'native_release_ack':self.current_release,
            'actual_current_checkpoint_pointer':self.current_capture['pointer'],'actual_owner':self.producer,
            'configuration':self.config_pin,'resident_native_boundary':self.current_idle if self.gpu is not None else None,
            'closed_native':self.closed_native if self.gpu is None else None,
            'current_exact_recovery':self.current_ack if self.gpu is None else None,
            'publication_cadence_evidence':self.current_cadence}
        if self.gpu is not None:
            journal=read_bound(self.current_capture['pointer'],self.parent.check)['journal_prefix']
            with Path(journal['path']).open('rb') as stream:creation=self.store._creation_identity(stream)
            value['incremental_authoritative_journal']={'kind':'ASR18_current_sole_append_journal_writer_identity_v1',
                'writer_source':self.config['journal_writer_source'],'path':journal['path'],'creation_identity':creation,
                'sole_current_writer':True,'append_operation':'open_ab_write_flush_fsync_only',
                'prior_bytes_never_rewritten_or_truncated':True,'writer_epoch':{
                    'kind':'ASR18_verified_continuous_append_writer_epoch_v1','writer_source':self.config['journal_writer_source'],
                    'producer_PID':self.gpu.producer['pid'],'producer_create_time':self.gpu.producer['create_time'],
                    'session_owner_nonce':self.current_worker_nonce,'configuration':self.current_worker_config}}
        if extra:value.update(extra)
        self.proof_sequence+=1;path=self.control/(self.current_intent['id']+'.proof.'+str(self.proof_sequence)+'.json')
        self.current_owner_proof=self._new(path,value);return self.current_owner_proof

    def verify_current_owner(self,request):
        self.check();require(request['proof']==self.current_owner_proof and request['ownership']==self.store.marker_pin,
            'Exact current source-bound owner operation witness')
        proof=read_bound(request['proof'],self.parent.check);release=read_bound(proof['native_release_ack'],self.parent.check)
        require(proof['actual_owner']==self.producer and proof['configuration']==self.config_pin
            and proof['actual_current_checkpoint_pointer']==self.current_capture['pointer'] and release['checkpoint_pointer']==self.current_capture['pointer']
            and release['request_nonce']==self.current_response['nonce'],'No stale/foreign current release witness')
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        if self.gpu is not None:
            require(release['producer']==self.gpu.producer and self.gpu.command_state=='coherent_idle_CPU_preparation',
                    'Actual continuous sole native writer is coherent idle')
            verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,phase=self.current_packet['phase'],
                proof_binding=self.current_idle,guard=self.parent.check)
            witness=proof['incremental_authoritative_journal'];epoch=witness['writer_epoch']
            require(witness['path']==pointer['journal_prefix']['path'] and epoch['configuration']==self.current_worker_config
                and epoch['session_owner_nonce']==self.current_worker_nonce and epoch['producer_PID']==self.gpu.producer['pid']
                and epoch['producer_create_time']==self.gpu.producer['create_time']
                and witness['writer_source']==record(Path(__file__).with_name('bpe_bounded_production_journal_v4.py')),
                'Exact source-bound live append writer, not mere saved metadata')
        else:
            require(proof['current_exact_recovery']==self.current_ack and proof['closed_native']==self.closed_native,
                    'No live native writer during current-recovered qualification')
            self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
        return True

    def _wait_ready(self,path):
        began=time.monotonic()
        while not path.exists():
            self.gpu.check();require(time.monotonic()-began<self.config['maximum_GPU_child_wall_seconds'],'Actual worker ready timeout');time.sleep(.1)
        return read_json(path)

    def _prepare(self,plan,index):
        self.check();intent=self.store.reserve_publication(publication_role='completed_window_end');self.current_intent=intent
        pp=self.control/(intent['id']+'.plan.json');require(not pp.exists(),'One fresh exact owned control plan')
        write_json(pp,plan);plan_pin=record(pp)
        workspace=self.store.reserve_auxiliary_workspace(intent,plan_pin);root=Path(workspace['workspace'])
        copied=root/'coordinator_plan_copy.json';shutil.copyfile(pp,copied)
        require(record(pp)['sha256']==record(copied)['sha256'] and pp.stat().st_size==copied.stat().st_size,'Exact complete source-owned plan bytecopy')
        write_json(root/'plan_alias.json',{'kind':'ASR18_exact_owned_control_plan_archive_alias_v1','publication_id':intent['id'],'original':plan_pin,'copied':record(copied)})
        if index>0:
            self._refresh_boundary()
            self.prior_boundary=self._new(root/'prior_window_boundary.json',self.boundary_state)
        bridge=bridge_plan(plan_pin,DATA_evidence=self.config['planning_arguments']['DATA_evidence'],session_binding=self.config['session'],
            window_index=index,prior_window_boundary=self.prior_boundary,verify_prior_boundary=self.verify_prior_boundary,guard=self.parent.check)
        bp=root/'bridge.json';write_json(bp,bridge);ba=root/'bridge_binding.json';write_json(ba,record(bp))
        cpu=root/'CPU';proof=self.run_CPU(self.config,self.config['CPU_controller_source']['path'],['--bridge',str(ba),'--output',str(cpu/'payload')],
            cpu,guard=self.parent.check if self.gpu is None else self.gpu.check,report_relative='payload/report.json')
        report=read_json(cpu/'payload/report.json');require(report['passed'] is True and report['all_children_actually_closed0'] is True,'Actual CPU8 full preparation0 before native')
        packet={'window_index':index,'phase':plan['phase'],'coordinator_plan':plan_pin,'feature_owner':report['owner'],'feature_manifest':report['manifest'],
            'session':self.config['session'],'lifecycle_parent_head':self.durable.head_pin,'checkpoint_intent':intent,'auxiliary_workspace':workspace}
        self.current_packet=packet;self.current_CPU0=proof;self.current_CPU_report=record(cpu/'payload/report.json');self.current_bridge=record(bp)
        self.current_workspace=workspace;self.current_original_plan=plan_pin
        return packet

    def verify_prior_boundary(self,boundary):
        self.check();require(self.prior_boundary is not None and boundary==read_bound(self.prior_boundary,self.parent.check)
            and boundary['durable_head']==self.durable.head_pin,'Exact previous owned coherent/cleanup/durable authority')
        return True

    def _start_native(self,packet):
        manifest=read_bound(packet['feature_manifest'],self.parent.check)
        require(self.native_sequence<513,'Bounded actual native restart/branch quota; no unchanged retry')
        self.native_sequence+=1;self.current_worker_nonce=secrets.token_hex(32);self.native_started=time.monotonic()
        worker={**self.config,'worker_mode':'resident_training','first_feature_update':manifest['updates'][0]['update'],
            'journal_namespace':str(self.root/'journals'),'initial_journal_path':str(self.initial_journal),
            'journal_resume_sequence':self.native_sequence,'session_owner_nonce':self.current_worker_nonce,
            'restore_parent_pointer':None if self.current_capture is None else self.current_capture['pointer']}
        if self.current_capture is not None:
            require(self.current_ack is not None and self.closed_native is not None,'No restart before exact current recovery')
            worker.update(source_native_process0=self.closed_native,source_checkpoint_capture=self.current_capture)
        path=self.control/('native_configuration_'+str(self.native_sequence).zfill(6)+'.json')
        self.current_worker_config=self._new(path,worker)
        self.gpu=self.GPUService.launch(path,self.folder/('native_'+str(self.native_sequence).zfill(6)),self.parent);self.parent.GPU=self.gpu
        ready=self._wait_ready(self.gpu.folder/'worker_ready.json')
        require(ready['optimizer_updates']==self.schedule.committed_updates(),
                'Original0 or actual current all-state startup restore BEFORE next draw')

    def _cleanup(self,completion,*,native0=None):
        pointer=read_bound(self.current_capture['pointer'],self.parent.check);CPU=read_json(self.current_CPU_report['path'])
        root=Path(self.current_workspace['workspace']);request={'owner':self.current_packet['feature_owner'],
            'closures':CPU['closures'],'manifest':self.current_packet['feature_manifest'],'parent_source':self.parent_source,
            'archive_path':str(root/'retained_feature_metadata.json'),'boundary':{
                'kind':'coherent_checkpoint_boundary','owner_sha256':self.current_packet['feature_owner']['sha256'],
                'session':self.config['session'],'phase':self.current_packet['phase'],
                'committed_window_updates':pointer['committed_window_updates'],
                'window_manifest_sha256':self.current_packet['feature_manifest']['sha256'],
                'checkpoint':pointer['checkpoint'],'coherent_checkpoint_pointer':self.current_capture['pointer']}}
        request['boundary']['resident_native_boundary' if native0 is None else 'actual_native_closed']=self.current_idle if native0 is None else native0
        path=root/'cleanup_request.json';write_json(path,request);out=root/'cleanup'
        proof=self.run_CPU(self.config,self.cleanup_script,['--cleanup','--request',str(path),'--folder',str(out)],out,
            guard=self.parent.check if self.gpu is None else self.gpu.check)
        result=read_json(out/'report.json');require(result['passed'] is True,'Actual exact owned featurecleanup0')
        self.current_cleanup0=proof;self.current_cleanup=record(out/'report.json')
        return result

    def _archive_and_publish(self,plan,completion):
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        metadata=read_bound(pointer['checkpoint'],self.parent.check);sampler=decode_scalar_tree(select(metadata['tree'],'sampler'))
        last=plan['updates'][-1];self.current_cadence={'kind':'ASR18_source_bound_actual_completed_window_cadence_v1',
            'phase':plan['phase'],'committed_window_updates':len(plan['updates']),'exact_pass_tail':bool(last['TR_draws'][-1]['draw_index']+1==
                (last['TR_draws'][-1]['epoch']+1)*(sampler['broad_cursor']['population'] if plan['phase']==WEAK else sampler['cleaner_cursor']['population'])),
            'coordinator_plan':self.current_original_plan}
        workspace=Path(self.current_workspace['workspace']);rows=[]
        with self.store.locked():inventory=self.store._workspace_inventory(self.current_workspace)
        directories=inventory['directories']
        for path in sorted(inventory['files'].values()):
            self.parent.check()
            if path==Path(self.current_workspace['workspace_marker']['path']):continue
            require(path.suffix.lower() in ('.json','.jsonl','.log','.txt','.sha256'),
                    'Reject unexpected audio/tensor/model BEFORE any full file hash')
            role='CPU_provenance'
            if path.name=='actual_pre_save_fingerprint.json':role='pre_save'
            elif path.name=='actual_next_draw_packet.json':role='next_draw'
            elif path.name=='persistence_verified.json':role='persistence'
            elif path.name in ('coordinator_plan_copy.json','plan_alias.json'):role='window_plan'
            elif path.name=='completion.json':role='native_completion'
            rows.append({'role':role,'file':record(path)})
        rows.sort(key=lambda row:Path(row['file']['path']).relative_to(workspace).as_posix())
        manifest={'kind':'ASR18_future_owned_window_auxiliary_manifest_v1','workspace_marker':self.current_workspace['workspace_marker'],
            'coordinator_plan':self.current_original_plan,'run_identity_sha256':digest(self.identity),
            'tensor_payloads_audio_features_and_models_included':False,'files':rows,'directories':sorted(directories),'journal_prefix':pointer['journal_prefix']}
        require(len(rows)<=self.store.policy['maximum_auxiliary_files'] and len(canonical(manifest))<=self.store.policy['maximum_auxiliary_header_bytes'],
            'New plan copy/alias fully counted, no silent archive cap expansion')
        self.schedule.load_state_dict(normalize_schedule_state(sampler))
        extra={'auxiliary_manifest_sha256':digest(manifest),'all_CPU_auxiliary_producers_actually_closed0':True,
            'native_auxiliary_writer_contexts_returned':True,'all_auxiliary_handles_released':True}
        with self.parent.metadata_phase('archive_aux'):
            proof=self._owner_proof(extra=extra)
            archived=self.store.archive_auxiliary(self.current_intent['id'],manifest,proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_auxiliary_archive_verified','archive':archived['archive']})
        expectation={'identity':pointer['identity'],'optimizer_updates':pointer['committed_updates'],'sampler_state':sampler,
            'journal_prefix':pointer['journal_prefix'],'run_identity':self.identity,'publication_role':'completed_window_end',
            'publication_cadence_evidence':self.current_cadence}
        with self.parent.metadata_phase('read_index'):
            proof=self._owner_proof();published=self.store.complete_publication(self.current_intent,self.current_capture['pointer'],expectation,proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_checkpoint_pair_publication_verified','archive':published['archive']})
            self.durable.record_pending_window(before_state=self.before_window,plan=self.current_original_plan,completion=completion)
        # Exact registered archive + immutable pending acknowledgment precede
        # retiring ONLY this newly created outside-store temporary control plan.
        with self.resolve_member({'kind':'ASR18_indexed_auxiliary_original_member_v1','original':self.current_original_plan,
                'storage_owner':self.store.marker_pin,'auxiliary_id':self.current_intent['id']}) as stream:
            h=hashlib.sha256();count=0
            while True:
                block=stream.read(1024**2)
                if not block:break
                h.update(block);count+=len(block)
        require(count==self.current_original_plan['bytes'] and h.hexdigest()==self.current_original_plan['sha256']
            and record(self.current_original_plan['path'])==self.current_original_plan,'Exact archive and actual owned original before single unlink')
        Path(self.current_original_plan['path']).unlink()
        self._refresh_boundary()
        self._retire_obsolete()

    def _recover(self):
        require(self.recovery_sequence<513,'Bounded coherent restart/current-recovery quota')
        self.recovery_sequence+=1;name='recovery_'+str(self.recovery_sequence).zfill(6)
        controls=self.control/name;controls.mkdir(exist_ok=False)
        self.closed_native=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        request={'kind':'ASR18_full_run_CURRENT_checkpoint_fresh_recovery_request_v1','identity':self.identity,
            'planning_arguments':self.config['planning_arguments'],'augmentation_identity':self.config['augmentation_identity'],
            'current_native_process0':self.closed_native,'current_checkpoint_capture':self.current_capture,
            'native_worker_source':self.config['worker_driver_source'],'native_parent_source':self.parent_source,
            'recovery_worker_source':self.config['worker_driver_source'],'recovery_parent_source':self.parent_source}
        path=controls/'request.json';self._new(path,request)
        config={**self.config,'worker_mode':'restore_only','current_recovery_request':record(path),
            'source_native_process0':self.closed_native,'source_checkpoint_capture':self.current_capture,'journal_namespace':str(self.root/'journals'),
            'journal_resume_sequence':self.recovery_sequence,'entire_worker_forecast':self.config['final_recovery_worker_forecast']}
        cp=controls/'worker_configuration.json';self._new(cp,config)
        self.gpu=self.GPUService.launch(cp,self.folder/name,self.parent);self.parent.GPU=self.gpu
        ready=self._wait_ready(self.gpu.folder/'current_recovery_ready.json');self.current_recovery_report=ready['report']
        self.closed_recovery=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        require(read_bound(self.closed_recovery,self.parent.check)['report']==self.current_recovery_report,'SAME canonical current recovery report in actualprocess0')
        evidence={'kind':'ASR18_full_run_current_recovery_prefix_inputs_v1','owner':OWNER,'diagnostic_only':False,'run_identity_sha256':digest(self.identity),
            'pending_head':self.durable.head_pin,'current_checkpoint_capture':self.current_capture,'current_native_process0':self.closed_native,
            'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report}
        ep=controls/'evidence.json';self._new(ep,evidence)
        request=self.durable._request(current_recovery_evidence=record(ep),coverage_output=str(self.root/'durable'/('coverage_'+str(self.recovery_sequence).zfill(6)+'.json')))
        with self.parent.metadata_phase('read_index'):
            coverage=create_coverage(request,resolve_member=self.resolve_member,guard=self.parent.check)
            self.current_coverage=coverage;v=read_bound(coverage,self.parent.check)
            ack={'kind':'ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1','diagnostic_only':False,
                'run_identity_sha256':digest(self.identity),'pending':deepcopy(self.durable.head['pending']),'pending_head':self.durable.head_pin,
                'final_checkpoint_pointer':self.current_capture['pointer'],'fresh_recovery_scope':'current_latest_only',
                'all_persistence_verified_through_pending_prefix':True,'all_historical_checkpoints_freshly_restored':False,
                'current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save':True,
                'actual_recovery_native_updates':0,'actual_recovery_draws_consumed':0,'actual_closed_native_index':self.closed_native,
                'actual_closed_recovery_index':self.closed_recovery,'persistence_prefix_coverage_index':coverage,
                'current_latest_exact_restore':v['current_latest_exact_restore'],'current_recovery_evidence':record(ep)}
            ap=controls/'acknowledgement.json';self.current_ack=self._new(ap,ack)
            self.durable.acknowledge_recovery(self.current_ack)
        self._refresh_boundary()

    def _refresh_boundary(self):
        if self.current_capture is None:return
        cleanup=read_bound(self.current_cleanup,self.parent.check)
        value={'kind':'FastMSS_actual_coherent_full_run_CPU_window_boundary_v2','session':self.config['session'],
            'completed_window_index':self.current_packet['window_index'],
            'bridge':self.current_bridge,'committed_window_updates':read_bound(self.current_capture['pointer'],self.parent.check)['committed_window_updates'],
            'optimizer_updates':self.schedule.committed_updates(),'sampler_state_sha256':digest(self.schedule.state_dict()),
            'coherent_checkpoint_pointer':self.current_capture['pointer'],'durable_head':self.durable.head_pin,
            'owned_cleanup_actual_closed':self.current_cleanup0,'owned_cleanup_receipt':self.current_cleanup,
            'preserved_metadata':cleanup['preserved_metadata'],'originals_checkpoints_and_journals_untouched':True,
            'owned_feature_window_retired':True}
        if self.gpu is None:value['actual_native_closed']=self.closed_native
        else:value['resident_native_boundary']=self.current_idle
        self.boundary_state=value

    def _retire_obsolete(self):
        # Bounded live B-tree sets only, never a Python full-history inventory.
        with self.parent.metadata_phase('read_index'):
            with self.store.locked():
                state=self.store._load()[0]
                protected=sorted(set(state['winners']+([state['latest']] if state['latest'] else [])))
                db=self.store._db()
                try:
                    live=[row[0] for row in db.execute('SELECT id FROM checkpoints WHERE retired=0 ORDER BY id LIMIT 6')]
                    aux=[row[0] for row in db.execute("SELECT id FROM auxiliary WHERE status='archived' ORDER BY id LIMIT 6")]
                finally:db.close()
                require(len(live)<=5,'Exact bounded B-tree live set; never iterate _Rows')
                for key in live:self.store._entry(key,verify_files=True)
                obsolete=[key for key in live if key not in protected]
            require(len(obsolete)<=5 and len(aux)<=5,'Bounded exact active owned sets')
            oldaux=[key for key in aux if key not in protected]
            if oldaux:
                proof=self._owner_proof(extra={'retire_auxiliary_ids':oldaux,'protected_active_auxiliary_ids':protected,
                    'all_auxiliary_handles_released':True})
                result=self.store.retire_obsolete_auxiliary(oldaux,proof,self.verify_current_owner)
                self._retire_witness(proof,{'kind':'exact_obsolete_auxiliary_retirement_returned','result':result})
            if obsolete:
                proof=self._owner_proof(extra={'retire_checkpoint_ids':obsolete})
                self.store.retire_obsolete_payloads(obsolete,proof,self.verify_current_owner)
                self._retire_witness(proof,{'kind':'exact_obsolete_payload_retirement_returned','checkpoint_ids':obsolete})

    def make_DEV_boundary(self,pointer):
        require(self.current_ack is not None and pointer==self.current_capture['pointer'] and self.gpu is None,'Actual CURRENT recovery before DEV')
        p=read_bound(pointer,self.parent.check)
        value={'kind':'ASR18_full_run_native_exit_and_CURRENT_checkpoint_recovery_boundary_for_serial_DEV_v2','owner':OWNER,'passed':True,
            'diagnostic_only':False,'source':self.parent_source,'production_recovery_scope':'CURRENT_LATEST_ONLY_WITH_VERIFIED_DURABLE_PERSISTENCE_PREFIX',
            'checkpoint_pointer':pointer,'run_identity_sha256':digest(self.identity),'optimizer_updates':p['committed_updates'],
            'sampler_state_sha256':p['sampler_state_sha256'],'journal_prefix':p['journal_prefix'],
            'all207_threeAdam_all4RNG_exact_checkpoint_restore_verified':True,'current_checkpoint_capture':self.current_capture,
            'current_native_process0':self.closed_native,'actual_recovery_process0':self.closed_recovery,'recovery_report':self.current_recovery_report,
            'production_recovery_boundary':self.current_ack,'durable_persistence_prefix_verification':self.current_coverage,'durable_head':self.durable.head_pin}
        path=self.control/('DEV_boundary_'+str(len(self.lifecycle.state_dict()['completed_boundaries']))+'.json');return self._new(path,value)

    def execute_DEV(self,request_pin):
        require(self.gpu is None and not self.config['owner_validation_only'],'Only actual declared pass-tail DEV, no validation fake barrier')
        from asr18_full_run_DEV_runtime_v4 import resource_proposal
        request=read_bound(request_pin,self.parent.check);name=request['action']['boundary']['name']
        folder=self.folder/('DEV_'+name)
        config={'kind':'ASR18_optimizer_free_serial350DEV_worker_configuration_v1','worker_mode':'serial350DEV_optimizer_free',
            'production_enabled':False,'diagnostic_only':False,'parent_driver_source':self.parent_source,
            'source_manifest':self.config['source_manifest'],'worker_driver_source':self.config['DEV_execution_sources']['worker_source'],
            'GPU_python':self.config['GPU_python'],'resource_policy':resource_proposal(),
            'entire_worker_forecast':{'entire_worker_private_forecast_bytes':5450000000},
            'base_assets':self.config['DEV_base_assets'],'source_bindings':self.config['DEV_source_bindings'],
            'request':request_pin,'owned_worker_admission_path':str(folder/'owned_worker_admission.json'),
            'ROOT_execution_review':self.config['ROOT_DEV_static_execution_review'],
            'ROOT_resource_review':self.config['ROOT_DEV_resource_review'],
            'maximum_GPU_child_wall_seconds':self.config['maximum_DEV_child_wall_seconds']}
        path=self.control/('DEV_configuration_'+name+'.json');self._new(path,config)
        self.gpu=self.GPUService.launch(path,folder,self.parent);self.parent.GPU=self.gpu
        actual0=self.gpu.finish();self.gpu=None;self.parent.GPU=None
        worker_report=read_bound(actual0,self.parent.check)['report']
        require(Path(worker_report['path'])==folder/'serial350_worker_report.json',
                'SAME actual autonomous inner report in process0; release envelope independently sealed')
        return {'worker_report':worker_report,'actual_process0':actual0}

    def _normal_budget_barrier(self):
        if self.gpu is None or self.current_capture is None:return False
        # Whole-window budget boundary only: never interrupt Adam or shorten a
        # pass. Preserve PRE/state/journal then actual0/current restore/prefix ACK.
        reserve=(self.config['maximum_CPU_phase_seconds']+self.config['maximum_native_window_seconds']
                 +self.config['maximum_archive_phase_seconds']+self.config['maximum_checkpoint_phase_seconds']
                 +self.config['maximum_native_exit_seconds'])
        require(0<reserve<self.config['maximum_GPU_child_wall_seconds'],
                'Reviewed native wall ceiling must support one complete next window plus closure')
        return time.monotonic()-self.native_started+reserve>=self.config['maximum_GPU_child_wall_seconds']

    def run(self):
        while True:
            self.check();require(time.monotonic()-self.began<self.config['maximum_owner_wall_seconds'],'Reviewed outer owner wall budget')
            if self._normal_budget_barrier():
                self._recover();continue
            action=self.durable.next_action()
            if action.get('kind')=='actual_native_exit_then_current_latest_recovery_and_persistence_prefix_required':
                self._recover();continue
            action=action.get('action',action)
            if action['kind']=='training_and_declared_DEV_complete':break
            if action['kind']=='checkpoint_then_serial350DEV':
                report=self.serial_DEV({'action':action,'checkpoint_pointer':self.current_capture['pointer'],'identity':self.identity,
                    'optimizer_updates':self.schedule.committed_updates(),'split':'validation','clips':350})
                from bpe_full_run_lifecycle_v2 import official_DEV
                score=official_DEV(read_bound(report,self.parent.check),self.current_capture['pointer'],read_bound(self.identity['scorer_policy'],self.parent.check))
                old=self.lifecycle.state_dict()['winner'];change=score['strict_all_six_passed'] and (old is None or score['word_errors']['tr']<old['word_errors']['tr'])
                winner=self.preserve_winner({'replace':change,'old_winner':old,'candidate_pointer':self.current_capture['pointer'],'DEV':report,'score':score})
                self.durable.acknowledge_DEV_and_winner(report=report,winner=winner);self._refresh_boundary();continue
            require(action['kind']=='bounded_training_window','Unsupported tail preserved, no guessed recovery')
            self.before_window=deepcopy(self.schedule.state_dict());count=len(self.lifecycle.state_dict()['completed_boundaries'])
            plan=(self.context.validation_plan(self.schedule,completed_boundaries=count,maximum_updates=4)
                if self.config['owner_validation_only'] else self.context.plan(self.schedule,completed_boundaries=count))
            packet=self._prepare(plan,self.last_window_index+1)
            if self.gpu is None:self._start_native(packet)
            response=self.gpu.command('run_prepared_window',window=packet);self.current_response=response
            self.current_completion=response['window_completion'];completion=read_bound(self.current_completion,self.parent.check)
            self.current_capture=completion['checkpoints'][0];self.current_release=response['checkpoint_storage_release_ack']
            self.current_idle=self.gpu.seal_idle(response)
            self._cleanup(completion)
            self._archive_and_publish(plan,self.current_completion);self.last_window_index+=1
            if self.config['owner_validation_only']:
                require(self.schedule.committed_updates()==4,'Bounded validation uses SAME original productionbody, exactstop4')
                self._recover();break
            self._refresh_boundary()

        if self.gpu is not None:self._recover()
        # Actually reopen the packed authority after current exact recovery.
        restored=self.lifecycle.state_dict();self.durable.close()
        lifecycle=FullRunLifecycle(self.schedule,identity=self.identity,callbacks=self.lifecycle.callbacks,
            qualified_winner=self.config['qualified_source_winner'],restored=restored,restore_proof=self.current_ack)
        self.lifecycle=lifecycle
        self.durable=PackedDurableLifecycleAdapter(lifecycle,self.root/'durable',policy=self.config['durable_policy'],proofs=self.proofs,
            create=False,resume_proof=self.current_ack)
        require(self.durable.state_dict()['pending']['publications']==0,'Actual current recovery and durable reopen0')
        result={'kind':'ASR18_actual_full_run_owner_pipeline_validation_v1','passed':True,'producer':self.producer,
            'worker_source':record(__file__),'parent_source':self.parent_source,'run_identity_sha256':digest(self.identity),
            'actual_optimizer_updates':self.schedule.committed_updates(),'completed_passes':len(self.lifecycle.state_dict()['completed_boundaries']),
            'current_checkpoint':self.current_capture,'current_native_process0':self.closed_native,'current_recovery_process0':self.closed_recovery,
            'current_recovery_report':self.current_recovery_report,'current_persistence_prefix_coverage':self.current_coverage,
            'durable_current_recovery_ack':self.current_ack,'durable_reopened_head':self.durable.head_pin,
            'qualified_source_winner_preserved':self.lifecycle.state_dict()['winner']==self.config['qualified_source_winner'],
            'owner_validation_only':self.config['owner_validation_only'],'production_source_eligible':not self.config['owner_validation_only'],
            'no_fake_pass_or_DEV_barrier':True,'production_enabled':False,'wall_seconds':time.monotonic()-self.began}
        write_json(self.folder/'report.json',result);return result

    def close(self):
        if getattr(self,'gpu',None) is not None:self.gpu.abort();self.gpu=None;self.parent.GPU=None
        if hasattr(self,'store'):self.store.close()
        if hasattr(self,'durable'):self.durable.close()
        if hasattr(self,'context'):self.context.close()
        if hasattr(self,'parent'):self.parent.close()


def run(config_path,review_path,folder,**services):
    owner=object.__new__(FullOwner)
    try:
        FullOwner.__init__(owner,config_path,review_path,folder,**services);return owner.run()
    finally:
        owner.close()
