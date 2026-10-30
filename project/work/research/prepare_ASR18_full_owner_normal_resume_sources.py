"""NEW normal coherent pause/resume body; no adoption of prior validation runs."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
old=S/'asr18_full_run_parent_controller_v11.py';new=S/'asr18_full_run_parent_controller_v12.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
check='''def checked_normal_resume_request(config_pin,review,resume_pin,*,parent_source,guard):
    """Pure bounded metadata/process checks BEFORE namespace mutation or Popen.

    Only a normally closed owner whose actual TOOL0 was independently observed
    may be resumed. Unknown suffixes, inflight work and live old owners stop.
    """
    from fastmss_owned_process_closure_v2 import dead
    from asr18_full_run_serial_DEV_v8 import read_logical_durable_head
    c=read_bound(config_pin,guard);request=read_bound(resume_pin,guard)
    require(review['resume_request']==resume_pin and review['invocation']=='normal_boundary_resume'
        and request['kind']=='ASR18_exact_normal_boundary_owner_resume_request_v1'
        and request['configuration']==config_pin and request['source_manifest']==c['source_manifest']
        and request['parent_source']==parent_source==c['parent_driver_source']
        and request['controller_source']==record(__file__)
        and request['run_identity_sha256']==digest(c['full_run_identity']),
        'Exact separately reviewed normal-resume request/source/configuration')
    snapshot=read_bound(request['normal_pause_snapshot'],guard)
    prior=read_bound(request['prior_owner_report'],guard);actual=read_bound(request['prior_owner_actual_TOOL0'],guard)
    require(snapshot['kind']=='ASR18_normal_coherent_owner_pause_snapshot_v1'
        and snapshot['configuration']==config_pin and snapshot['run_identity_sha256']==digest(c['full_run_identity'])
        and snapshot['parent_source']==parent_source and snapshot['controller_source']==record(__file__)
        and snapshot['namespace']==c['run_namespace'] and snapshot['normal_coherent_pause'] is True,
        'Same immutable genesis identity, no historical namespace adoption')
    require(prior['kind']=='ASR18_full_owner_normal_boundary_pause_report_v1' and prior['passed'] is True
        and prior['normal_pause_snapshot']==request['normal_pause_snapshot'] and prior['producer']==snapshot['producer']
        and prior['configuration']==config_pin and actual['kind']=='ASR18_observed_normal_owner_pause_TOOL0_v1'
        and actual['actual_tool_exit_code']==0 and actual['report']==request['prior_owner_report']
        and actual['normal_pause_snapshot']==request['normal_pause_snapshot']
        and actual['producer']==prior['producer'] and actual['parent_source']==parent_source
        and actual['configuration']==config_pin and dead(prior['producer']),
        'Actual normal owner TOOL0 and exact old producer dead, never file-existence inference')
    for key in ('closed_native','closed_recovery'):
        proof=read_bound(snapshot[key],guard);verify(proof,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,guard=guard)
        require(proof['parent_source']==parent_source and proof['worker_source']==c['worker_driver_source'],
                'Same genesis native/recovery execution sources')
    pointer=read_bound(snapshot['current_capture']['pointer'],guard)
    ack=read_bound(snapshot['current_ack'],guard);coverage=read_bound(snapshot['current_coverage'],guard)
    require(ack['kind']=='ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1'
        and ack['final_checkpoint_pointer']==snapshot['current_capture']['pointer']
        and ack['current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save'] is True
        and ack['actual_closed_native_index']==snapshot['closed_native']
        and ack['actual_closed_recovery_index']==snapshot['closed_recovery']
        and ack['persistence_prefix_coverage_index']==snapshot['current_coverage']
        and coverage['passed'] is True and coverage['historical_fresh_GPU_restores_claimed'] is False,
        'Actual current restore and complete PRE/persistence prefix ACK')
    latest=read_json(Path(c['run_namespace'])/'durable/latest.json')
    require(latest['head']==snapshot['durable_head'] and latest['owner']==snapshot['durable_owner'],
            'No stale ACK/head, no newer durable state silently dropped')
    head=read_logical_durable_head(snapshot['durable_head'])['head']
    require(head['identity_sha256']==digest(c['full_run_identity']) and head['owner']==snapshot['durable_owner']
        and head['pending']['windows']==head['pending']['publications']==0
        and head['lifecycle']==snapshot['lifecycle_state']
        and head['lifecycle']['last_recovery']==snapshot['current_ack'],
        'Entire bounded lifecycle/winner and pending0 are exact packed authority')
    state=read_json(Path(c['checkpoint_storage_namespace'])/'state.json')
    require(state['owner']==snapshot['storage_owner'] and state['latest']==snapshot['current_intent']['id']
        and state['pending_publication'] is None and state['pending_retirement'] is None
        and state['catalog']==snapshot['storage_catalog'],
        'No pending publication/retirement/inflight workspace at normal pause')
    sampler=snapshot['sampler_state'];frontier=progress(sampler)
    require(digest(sampler)==pointer['sampler_state_sha256']==head['lifecycle']['last_sampler_sha256']
        and frontier['optimizer_updates']==pointer['committed_updates']==snapshot['actual_optimizer_updates']
        and frontier['optimizer_updates']==4 and frontier['broad_draws']==56 and frontier['cleaner_draws']==0,
        'Original first4 counter/sampler frontier, never a reset or guessed reconstruction')
    packet=read_bound(snapshot['current_capture']['next_draw_packet'],guard)
    require(digest(packet)==snapshot['next_draw_packet_sha256'],'Exact saved next-draw packet')
    prefix=pointer['journal_prefix'];path=Path(prefix['path']).resolve()
    require(path.parent==Path(c['run_namespace'])/'journals' and path.stat().st_size==prefix['offset']
        and prefix==snapshot['journal_prefix'],'Whole closed registered journal only; unknown suffix preserved/rejected')
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):guard();sha.update(block)
    require(sha.hexdigest()==prefix['sha256'],'Exact complete saved journal bytes')
    require(snapshot['native_sequence']==snapshot['recovery_sequence']==1 and snapshot['last_window_index']==0
        and snapshot['actual_owner_invocation_count']==1,'Exact first normal pause only in bounded split proof')
    return snapshot


'''
text=text.replace('class FullOwner:\n',check+'class FullOwner:\n',1)
text=text.replace('storage,fresh_admit,execute_owned_DEV,native_boundary_DEV,install_owner):','storage,fresh_admit,execute_owned_DEV,native_boundary_DEV,install_owner,resume_request=None):',1)
text=text.replace("self.native_sequence=0;self.recovery_sequence=0;self.proof_sequence=0;self.boundary_state=None",
    "self.native_sequence=0;self.recovery_sequence=0;self.proof_sequence=0;self.boundary_state=None\n        self.resume_request=resume_request;self.resume_snapshot=None;self.actual_owner_invocation_count=1")
text=text.replace("r['scope']=='finite_full_run_owner_eight_update_two_window_pipeline_validation_v1'",
    "r['scope']=='finite_full_owner_first4_normal_pause_resume_next4_equivalence_v1'")
before="""        require(self.folder.is_relative_to(ROOT) and self.root.is_relative_to(ROOT/'runs')
                and not self.folder.exists() and not self.root.exists(),'Fresh exact owner namespaces, never existing artifact adoption')
        self.folder.mkdir();self.root.mkdir();self.control.mkdir();self.parent=ParentGuard(self.folder)"""
after="""        require(c['normal_boundary_resume_enabled'] is True and c['owner_validation_pause_after_updates']==4,
                'Resume-enabled immutable genesis config required from first invocation')
        if resume_request is not None:
            self.resume_snapshot=checked_normal_resume_request(self.config_pin,r,resume_request,
                parent_source=parent_source,guard=lambda:None)
            require(self.root.is_dir() and self.control.is_dir(),'Only verified normal-pause namespace may reopen')
        else:
            require(r['invocation']=='genesis_normal_pause4' and not self.root.exists(),
                    'Explicit resume request required; no implicit existing namespace adoption')
        require(self.folder.is_relative_to(ROOT) and self.root.is_relative_to(ROOT/'runs')
                and not self.folder.exists(),'Fresh actual outer supervisor output folder per invocation')
        self.folder.mkdir()
        if self.resume_snapshot is None:self.root.mkdir();self.control.mkdir()
        self.parent=ParentGuard(self.folder)"""
assert text.count(before)==1;text=text.replace(before,after)
before="""        self.journal_namespace=JournalNamespace(self.root/'journals',identity=self.identity,guard=self.parent.check,create=True)
        self.initial_journal=self.journal_namespace.register('commits_000000.jsonl')"""
after="""        resuming=self.resume_snapshot is not None
        self.journal_namespace=JournalNamespace(self.root/'journals',identity=self.identity,guard=self.parent.check,create=not resuming)
        if resuming:
            snapshot=checked_normal_resume_request(self.config_pin,r,resume_request,parent_source=parent_source,guard=self.parent.check)
            self.schedule.load_state_dict(normalize_schedule_state(snapshot['sampler_state']))
            for key in ('current_capture','current_completion','current_release','current_recovery_report','current_ack',
                'current_coverage','current_packet','current_CPU0','current_CPU_report','current_bridge',
                'current_cleanup0','current_cleanup','current_workspace','current_original_plan','current_intent',
                'closed_native','closed_recovery','native_sequence','recovery_sequence','proof_sequence','last_window_index'):
                setattr(self,key,deepcopy(snapshot[key]))
            self.actual_owner_invocation_count=2;self.initial_journal=Path(snapshot['journal_prefix']['path'])
            require(self.initial_journal in self.journal_namespace.paths,'Same exact registered journal, no branch/copy/reset')
            require(digest(self.context.next_packet(self.schedule))==snapshot['next_draw_packet_sha256'],
                    'Actual source-context next draw matches normal pause BEFORE any prepare/Popen')
        else:self.initial_journal=self.journal_namespace.register('commits_000000.jsonl')"""
assert text.count(before)==1;text=text.replace(before,after)
before="""        self.lifecycle=FullRunLifecycle(self.schedule,identity=self.identity,callbacks=callbacks,
            qualified_winner=deepcopy(c['qualified_source_winner']))"""
after="""        self.lifecycle=FullRunLifecycle(self.schedule,identity=self.identity,callbacks=callbacks,
            qualified_winner=deepcopy(c['qualified_source_winner']),
            restored=None if not resuming else snapshot['lifecycle_state'],
            restore_proof=None if not resuming else self.current_ack)"""
assert text.count(before)==1;text=text.replace(before,after)
text=text.replace("policy=c['durable_policy'],proofs=proofs,create=True)","policy=c['durable_policy'],proofs=proofs,create=not resuming,\n                resume_proof=None if not resuming else self.current_ack)",1)
before="""            self.store=IndexedOwnedFutureCheckpointStore.create(c['checkpoint_storage_namespace'],c['checkpoint_storage_policy'],
                ROOT_review=c['ROOT_storage_review'],run_identity=self.identity,verifier_source=record(__file__),
                protected=c['protected_external'],guard=self.parent.check)"""
after="""            self.store=(IndexedOwnedFutureCheckpointStore(snapshot['storage_owner'],guard=self.parent.check) if resuming else
                IndexedOwnedFutureCheckpointStore.create(c['checkpoint_storage_namespace'],c['checkpoint_storage_policy'],
                    ROOT_review=c['ROOT_storage_review'],run_identity=self.identity,verifier_source=record(__file__),
                    protected=c['protected_external'],guard=self.parent.check))"""
assert text.count(before)==1;text=text.replace(before,after)
text=text.replace("'run_identity_sha256':digest(self.identity),'production_enabled':False})\n\n    def _new",
    "'run_identity_sha256':digest(self.identity),'resume_request':resume_request,\n            'actual_owner_invocation_count':self.actual_owner_invocation_count,'production_enabled':False})\n\n    def _new",1)
text=text.replace("        path=self.root/'DEV';path.mkdir(exist_ok=False);return path",
    "        path=self.root/'DEV'\n        if self.resume_snapshot is None:path.mkdir(exist_ok=False)\n        else:require(path.is_dir(),'Same owned DEV output root on explicit normal resume')\n        return path",1)
text=text.replace("and digest(request['schedule'])==self.lifecycle.state_dict()['last_sampler_sha256'],'Current authoritative lifecycle/schedule')",
    "and digest(request['schedule'])==request['lifecycle']['last_sampler_sha256'],'Current authoritative lifecycle/schedule')",1)
pause='''    def _normal_owner_pause(self):
        require(self.actual_owner_invocation_count==1 and self.schedule.committed_updates()==4,
                'One explicit normally coherent first4 pause, no crash-tail recovery')
        self._recover()
        require(self.gpu is None and self.durable.state_dict()['pending']['publications']==0,
                'Actual native0/CURRENT recovery0/prefix ACK before normal owner pause')
        state=self._read_state();require(state['pending_publication'] is None and state['pending_retirement'] is None,
                'No storage mutation in flight at normal owner pause')
        pointer=read_bound(self.current_capture['pointer'],self.parent.check)
        fields=('current_capture','current_completion','current_release','current_recovery_report','current_ack',
            'current_coverage','current_packet','current_CPU0','current_CPU_report','current_bridge',
            'current_cleanup0','current_cleanup','current_workspace','current_original_plan','current_intent',
            'closed_native','closed_recovery','native_sequence','recovery_sequence','proof_sequence','last_window_index')
        value={key:deepcopy(getattr(self,key)) for key in fields}
        value.update(kind='ASR18_normal_coherent_owner_pause_snapshot_v1',configuration=self.config_pin,
            run_identity_sha256=digest(self.identity),namespace=str(self.root),producer=self.producer,
            parent_source=self.parent_source,controller_source=record(__file__),normal_coherent_pause=True,
            actual_optimizer_updates=4,actual_owner_invocation_count=1,
            sampler_state=deepcopy(self.schedule.state_dict()),lifecycle_state=deepcopy(self.lifecycle.state_dict()),
            durable_head=deepcopy(self.durable.head_pin),durable_owner=self.durable.owner,storage_owner=self.store.marker_pin,
            storage_catalog=deepcopy(state['catalog']),journal_prefix=deepcopy(pointer['journal_prefix']),
            next_draw_packet_sha256=digest(read_bound(self.current_capture['next_draw_packet'],self.parent.check)))
        snapshot=self._new(self.control/'normal_pause_000001.json',value)
        report={'kind':'ASR18_full_owner_normal_boundary_pause_report_v1','passed':True,'producer':self.producer,
            'worker_source':record(__file__),'parent_source':self.parent_source,'configuration':self.config_pin,
            'run_identity_sha256':digest(self.identity),'actual_optimizer_updates':4,'normal_pause_snapshot':snapshot,
            'normal_internal_native_and_recovery_processes_actually_closed0':True,
            'outer_actual_TOOL0_still_must_be_observed':True,'completed_passes':0,'production_enabled':False}
        write_json(self.folder/'report.json',report);return report

'''
text=text.replace('    def run(self):\n',pause+'    def run(self):\n',1)
before="""                if actual==8:
                    self._recover();break"""
after="""                if actual==4 and self.actual_owner_invocation_count==1:return self._normal_owner_pause()
                if actual==8:
                    self._recover();break"""
assert text.count(before)==1;text=text.replace(before,after)
text=text.replace("'actual_optimizer_updates':self.schedule.committed_updates(),'completed_passes':",
    "'actual_owner_invocation_count':self.actual_owner_invocation_count,'actual_normal_resume_request':self.resume_request,\n            'actual_optimizer_updates':self.schedule.committed_updates(),'completed_passes':",1)
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
print(json.dumps({'draft_controller':str(new),'bytes':new.stat().st_size,
    'sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'source_only_no_target_import':True,
    'pending_worker_pool_parent_API_before_freeze':True}))
