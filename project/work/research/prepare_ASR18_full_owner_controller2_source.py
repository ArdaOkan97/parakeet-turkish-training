"""Create NEW controller2 source; stdlib AST only, no target imports/jobs."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
old=S/'asr18_full_run_parent_controller_v1.py';new=S/'asr18_full_run_parent_controller_v2.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
def rep(a,b):
    global text
    assert text.count(a)==1,a[:120];text=text.replace(a,b)
def method(name,body):
    global text
    start=text.index('    def '+name+'(')
    ends=[v for v in (text.find('\n    def ',start+1),text.find('\n    @',start+1),text.find('\n\ndef ',start+1)) if v>=0]
    end=min(ends) if ends else len(text)
    text=text[:start]+body.rstrip()+'\n'+text[end:]

rep('from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state',
    'from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state\nfrom bpe_training_schedule_v1 import WEAK')
rep('self.current_packet=None;self.last_window_index=-1;self.resume_checked=False;self.DEV=None',
    'self.current_packet=None;self.last_window_index=-1;self.resume_checked=False;self.DEV=None\n        self.native_sequence=0;self.recovery_sequence=0;self.proof_sequence=0;self.boundary_state=None')
rep("    def _DEV_folder(self):",'''    def _new(self,path,value):
        path=Path(path).resolve()
        require((path.is_relative_to(self.root) or path.is_relative_to(self.folder)) and not path.exists(),
                'Fresh immutable exact owned control artifact')
        write_json(path,value);return record(path)

    def _retire_witness(self,pin,mutation):
        # Storage embeds the FULL original witness and pin in its verified
        # immutable archive/history. No consumer later reads this transient path.
        require(record(pin['path'])==pin,'Same returned-operation witness before exact owned retirement')
        path=Path(pin['path']).resolve()
        require(path.parent==self.control and '.proof.' in path.name,'Only new exact operation witness')
        event={'kind':'ASR18_exact_transient_storage_witness_retired_after_verified_mutation_v1',
            'original':pin,'storage_owner':self.store.marker_pin,'successful_mutation':mutation,
            'full_original_witness_embedded_and_verified_by_storage':True,'producer':self.producer}
        raw=canonical(event)+b'\\n';log=self.control/'transient_retirement.jsonl'
        require((log.stat().st_size if log.exists() else 0)+len(raw)<=256*1024**2,'Bounded compact retirement control log')
        with log.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        path.unlink();require(not path.exists(),'Actual exact transient witness retirement')

    def _DEV_folder(self):''')
method('verify_winner','''    def verify_winner(self,request):
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
        require(report is not None and artifact==self.current_capture['pointer'] and self.current_ack is not None,
                'Only current actual recovered pointer can become a new qualified winner')
        actual=read_bound(report,self.parent.check)
        source_request=read_bound(actual['source_request'],self.parent.check)
        from asr18_full_run_serial_DEV_v9 import verified_completed_DEV
        require(verified_completed_DEV(actual['source_request'],actual['actual_worker_report'],actual['actual_owned_DEV_process0'])==actual,
                'Same sealed actual official350 execution, never a retention fixture')
        from bpe_full_run_lifecycle_v2 import official_DEV
        score=official_DEV(actual,artifact,read_bound(self.identity['scorer_policy'],self.parent.check))
        require(score['strict_all_six_passed'] and source_request['identity']==self.identity,
                'Exact integer all-six retention and original full-run identity')
        if type(winner) is dict and 'artifact' in winner:
            require(winner['strict_qualified'] and winner['word_errors']==score['word_errors']
                and winner['reference_words']==REFERENCES,'Stored winner exact official score')
        return True''')
method('preserve_winner','''    def preserve_winner(self,request):
        if not request['replace']:return request['old_winner']['artifact']
        require(request['score']['strict_all_six_passed'] and self.gpu is None and self.current_ack is not None,
                'Actual current fresh recovery and official strict qualification before winner protection')
        self.verify_winner({'winner':request['candidate_pointer'],'DEV':request['DEV']})
        with self.parent.metadata_phase('read_index'):
            proof=self._owner_proof(extra={'qualification':request['DEV'],'strict_all_six_qualified':True,
                'qualified_pointer':self.current_capture['pointer']})
            self.store.protect_qualified_winner(self.current_intent['id'],proof,self.verify_current_owner)
            self._retire_witness(proof,{'kind':'actual_qualified_winner_protected','checkpoint_id':self.current_intent['id']})
        return request['candidate_pointer']''')
method('_owner_proof','''    def _owner_proof(self,*,extra=None):
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
        self.current_owner_proof=self._new(path,value);return self.current_owner_proof''')
method('verify_current_owner','''    def verify_current_owner(self,request):
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
        return True''')
rep("        bridge=bridge_plan(plan_pin,DATA_evidence=", "        if index>0:\n            self._refresh_boundary()\n            self.prior_boundary=self._new(root/'prior_window_boundary.json',self.boundary_state)\n        bridge=bridge_plan(plan_pin,DATA_evidence=")
method('_start_native','''    def _start_native(self,packet):
        manifest=read_bound(packet['feature_manifest'],self.parent.check)
        require(self.native_sequence<513,'Bounded actual native restart/branch quota; no unchanged retry')
        self.native_sequence+=1;self.current_worker_nonce=secrets.token_hex(32)
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
                'Original0 or actual current all-state startup restore BEFORE next draw')''')
rep("'archive_path':str(root/'retained_feature_metadata.json'),'boundary':{'phase':self.current_packet['phase'],\n                'coherent_checkpoint_pointer':self.current_capture['pointer']}}",
"'archive_path':str(root/'retained_feature_metadata.json'),'boundary':{\n                'kind':'coherent_checkpoint_boundary','owner_sha256':self.current_packet['feature_owner']['sha256'],\n                'session':self.config['session'],'phase':self.current_packet['phase'],\n                'committed_window_updates':pointer['committed_window_updates'],\n                'window_manifest_sha256':self.current_packet['feature_manifest']['sha256'],\n                'checkpoint':pointer['checkpoint'],'coherent_checkpoint_pointer':self.current_capture['pointer']}}")
rep("plan['phase']=='weak_data_training'","plan['phase']==WEAK")
rep("        workspace=Path(self.current_workspace['workspace']);rows=[];directories=['.']\n        for path in sorted(workspace.rglob('*')):\n            self.parent.check()\n            if path.is_dir():directories.append(path.relative_to(workspace).as_posix());continue\n            if path==Path(self.current_workspace['workspace_marker']['path']):continue",
"        workspace=Path(self.current_workspace['workspace']);rows=[]\n        with self.store.locked():inventory=self.store._workspace_inventory(self.current_workspace)\n        directories=inventory['directories']\n        for path in sorted(inventory['files'].values()):\n            self.parent.check()\n            if path==Path(self.current_workspace['workspace_marker']['path']):continue\n            require(path.suffix.lower() in ('.json','.jsonl','.log','.txt','.sha256'),\n                    'Reject unexpected audio/tensor/model BEFORE any full file hash')")
rep("            self.store.archive_auxiliary(self.current_intent['id'],manifest,proof,self.verify_current_owner)",
"            archived=self.store.archive_auxiliary(self.current_intent['id'],manifest,proof,self.verify_current_owner)\n            self._retire_witness(proof,{'kind':'actual_auxiliary_archive_verified','archive':archived['archive']})")
rep("            proof=self._owner_proof();self.store.complete_publication(self.current_intent,self.current_capture['pointer'],expectation,proof,self.verify_current_owner)",
"            proof=self._owner_proof();published=self.store.complete_publication(self.current_intent,self.current_capture['pointer'],expectation,proof,self.verify_current_owner)\n            self._retire_witness(proof,{'kind':'actual_checkpoint_pair_publication_verified','archive':published['archive']})")
rep("        Path(self.current_original_plan['path']).unlink()", "        Path(self.current_original_plan['path']).unlink()\n        self._refresh_boundary()\n        self._retire_obsolete()")
rep("    def _recover(self):\n        self.closed_native=self.gpu.finish();self.gpu=None;self.parent.GPU=None",
"    def _recover(self):\n        require(self.recovery_sequence<513,'Bounded coherent restart/current-recovery quota')\n        self.recovery_sequence+=1;name='recovery_'+str(self.recovery_sequence).zfill(6)\n        controls=self.control/name;controls.mkdir(exist_ok=False)\n        self.closed_native=self.gpu.finish();self.gpu=None;self.parent.GPU=None")
rep("path=self.control/'current_recovery_request.json';write_json(path,request)","path=controls/'request.json';self._new(path,request)")
rep("'journal_resume_sequence':1,'entire_worker_forecast':self.config['final_recovery_worker_forecast']}","'journal_resume_sequence':self.recovery_sequence,'entire_worker_forecast':self.config['final_recovery_worker_forecast']}")
rep("cp=self.control/'current_recovery_worker_configuration.json';write_json(cp,config)","cp=controls/'worker_configuration.json';self._new(cp,config)")
rep("self.gpu=self.GPUService.launch(cp,self.folder/'recovery',self.parent)","self.gpu=self.GPUService.launch(cp,self.folder/name,self.parent)")
rep("evidence={'kind':'ASR18_full_run_current_recovery_prefix_inputs_v1','run_identity_sha256':digest(self.identity)",
"evidence={'kind':'ASR18_full_run_current_recovery_prefix_inputs_v1','owner':OWNER,'diagnostic_only':False,'run_identity_sha256':digest(self.identity)")
rep("ep=self.control/'current_recovery_evidence.json';write_json(ep,evidence)","ep=controls/'evidence.json';self._new(ep,evidence)")
rep("coverage_output=str(self.control/'current_persistence_prefix_coverage.json')", "coverage_output=str(self.root/'durable'/('coverage_'+str(self.recovery_sequence).zfill(6)+'.json'))")
rep("ap=self.control/'current_recovery_acknowledgement.json';write_json(ap,ack);self.current_ack=record(ap)","ap=controls/'acknowledgement.json';self.current_ack=self._new(ap,ack)")
rep("            self.durable.acknowledge_recovery(self.current_ack)","            self.durable.acknowledge_recovery(self.current_ack)\n        self._refresh_boundary()")
rep("path=self.control/'current_DEV_native_boundary.json';write_json(path,value);return record(path)",
"path=self.control/('DEV_boundary_'+str(len(self.lifecycle.state_dict()['completed_boundaries']))+'.json');return self._new(path,value)")
method('execute_DEV','''    def execute_DEV(self,request_pin):
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
        return {'worker_report':worker_report,'actual_process0':actual0}''')
rep("self.durable.acknowledge_DEV_and_winner(report=report,winner=winner);continue",
"self.durable.acknowledge_DEV_and_winner(report=report,winner=winner);self._refresh_boundary();continue")
start=text.index('            # Next window is prepared from this exact saved pointer/durablehead,')
end=text.index('\n        if self.gpu is not None:self._recover()',start)
text=text[:start]+"            self._refresh_boundary()\n"+text[end:]
rep("    def make_DEV_boundary(self,pointer):",'''    def _refresh_boundary(self):
        if self.current_capture is None:return
        cleanup=read_bound(self.current_cleanup,self.parent.check)
        value={'kind':'FastMSS_actual_coherent_full_run_CPU_window_boundary_v2','session':self.config['session'],
            'completed_window_index':self.last_window_index if self.last_window_index>=0 else self.current_packet['window_index'],
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
                state,entries,_=self.store._load()
                protected=sorted(set(state['winners']+([state['latest']] if state['latest'] else [])))
                obsolete=[key for key,entry in entries.items() if key not in protected and not entry.get('retired')]
                db=self.store._db()
                try:aux=[row[0] for row in db.execute("SELECT id FROM auxiliary WHERE status='archived' ORDER BY id LIMIT 6")]
                finally:db.close()
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

    def make_DEV_boundary(self,pointer):''')
rep("            'completed_window_index':self.last_window_index if self.last_window_index>=0 else self.current_packet['window_index'],",
"            'completed_window_index':self.current_packet['window_index'],")
rep("        self.context.close();self.parent.close()", "        if hasattr(self,'context'):self.context.close()\n        if hasattr(self,'parent'):self.parent.close()")
rep("    owner=None\n    try:\n        owner=FullOwner(config_path,review_path,folder,**services);return owner.run()\n    finally:\n        if owner is not None:owner.close()",
"    owner=object.__new__(FullOwner)\n    try:\n        FullOwner.__init__(owner,config_path,review_path,folder,**services);return owner.run()\n    finally:\n        owner.close()")
rep("        if self.gpu is not None:self.gpu.abort();self.gpu=None;self.parent.GPU=None", "        if getattr(self,'gpu',None) is not None:self.gpu.abort();self.gpu=None;self.parent.GPU=None")
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
