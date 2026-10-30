"""Resident CPU/RAM/native continuation of the exact saved training state.

One GPU owner, private CPU planning, real positive-Adam journal commits, full
streamed checkpoints every 200 updates and at declared pass/DEV boundaries.
Feature arrays never become files. Old RUN07 files are read, never rewritten.
"""
import argparse,copy,gc,gzip,hashlib,inspect,json,os,secrets,shutil,sys,threading,time,traceback
from contextlib import contextmanager,ExitStack
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest,canonical
ROOT=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis'


def commit_RAM_transaction(tx,optimizer,journal,native,evidence):
    """Same scientific commit as ScheduledTransaction.commit, explicit RAM evidence."""
    schedule=tx.schedule
    require(not tx.closed and not tx.poisoned and not schedule.poisoned and schedule.state_dict()==tx.before,
        'Only current untouched sampler transaction commits')
    require(not optimizer.poisoned and optimizer.updates==tx.optimizer_updates+1,'Exactly one actual positive Adam')
    try:
        expected=tx.TR_identity+list(tx.replay_rows)
        require(native['update']==optimizer.updates and native['phase']==tx.phase
            and native['source_ids']==[r['source_id']for r in expected]
            and native['languages']==[r['language']for r in expected], 'Exact ordered source/phase native receipt')
        require(evidence['kind']=='ASR18_resident_RAM_preparation_commit_evidence_v1'
            and len(evidence['metadata_sha256'])==64 and evidence['actual_positive_native_update']==optimizer.updates,
            'Truthful consumed RAM packet, no fictitious closed file/process')
        tx.cursor.commit(tx.TR_draws);schedule.replay.load_state_dict(tx.after_replay);state=schedule.state_dict()
        compact={k:native[k]for k in ('update','phase','source_ids','languages','mean_objective','global_grad_norm')}
        compact.update(full_native_receipt_sha256=digest(native),augmentation_sha256=digest(native['augmentation']))
        journal.append({'kind':'ASR18_committed_training_update_v1','phase':tx.phase,'optimizer_updates':optimizer.updates,
            'transaction_sha256':tx.identity_digest,'sampler_state_sha256':digest(state),
            'broad_draws':schedule.broad.draws,'cleaner_draws':schedule.cleaner.draws,'replay_updates':schedule.replay.updates,
            'prepared_evidence':evidence,'native':compact})
        tx.closed=True;return state
    except BaseException:
        tx.poisoned=schedule.poisoned=optimizer.poisoned=True;raise


class SavedJournalPrefixView:
    """Readonly saved frontier while the real failed writer's later tail is retained."""
    def __init__(self,saved,guard):self.saved=copy.deepcopy(saved);self.guard=guard;self.verify(saved)
    def binding(self):return copy.deepcopy(self.saved)
    def verify(self,binding):
        from bpe_bounded_production_journal_v4 import prefix
        require(binding==self.saved and prefix(Path(binding['path']),binding['offset'],self.guard).hexdigest()==binding['sha256'],
            'Actual unchanged saved journal prefix; later tail is never replayed')
        return True
    def append(self,event):raise RuntimeError('Readonly checkpoint restore view cannot append')


def checked_headroom_resume(pin,*,base,folder,pointer_pin):
    from fastmss_owned_process_closure_v2 import dead
    require(pin==base['prior_resident_RAM_headroom_failure'] and record(pin['path'])==pin,'ROOT-bound actual resource failure')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_resident_RAM_headroom_failure_saved3060_v1'
        and value['actual_exec_tool_exit_code']==1 and value['normal_native0_claimed'] is False
        and value['intentional_stop_claimed'] is False and value['branch']==str(folder)
        and value['checkpoint_pointer']==pointer_pin and value['saved_optimizer_updates']==3060
        and value['actual_optimizer_updates_before_failure']==3069
        and all(dead(w)for w in value['captured_identities']),'Actual abnormal3069/dead owner and coherent saved3060')
    for key in ('original_failure','original_monitor_completion','original_CPU_stderr','whole_journal_including_preserved_tail','preserved_unsaved_tail'):
        require(record(value[key]['path'])==value[key],'Preserve exact failure and original positive tail')
    require(value['preserved_tail_positive_updates']==list(range(3061,3070)),'Nine actual unsaved positive updates stay historical')
    return value


def checked_DEV_frontend_resume(pin,*,base,folder,pointer_pin):
    from fastmss_owned_process_closure_v2 import dead
    require(pin==base['prior_resident_RAM_DEV_frontend_failure'] and record(pin['path'])==pin,
        'Exact reviewed failed periodic frontend owner; no normal0/resource-stop relabel')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_resident_RAM_periodic_DEV_frontend_failure_saved4000_v1'
        and value['actual_exec_tool_exit_code']==1 and value['normal_native0_claimed'] is False
        and value['intentional_stop_claimed'] is False and value['branch']==str(folder)
        and value['checkpoint_pointer']==pointer_pin
        and value['saved_optimizer_updates']==value['actual_optimizer_updates_before_failure']==4000
        and value['unsaved_tail_updates']==[] and value['periodic4000_still_required_before4001'] is True
        and value['actual_WER_report_produced'] is False and all(dead(w)for w in value['captured_identities']),
        'Actual failed first-row audio frontend/dead identities and coherent4000')
    for key in ('original_failure','original_monitor_completion','original_CPU_stderr','whole_journal',
                'checkpoint_pointer','checkpoint_persistence','original_native_log','source_configuration'):
        require(record(value[key]['path'])==value[key],'Preserved source3 failure/saved state/wholejournal')
    for pin in value['failed_results'].values():require(record(pin['path'])==pin,'Old partial DEV predictions/results untouched')
    require(read_json(pointer_pin['path'])['journal_prefix']==value['saved_journal_prefix']
        and value['whole_journal']=={'path':value['saved_journal_prefix']['path'],
            'bytes':value['saved_journal_prefix']['offset'],'sha256':value['saved_journal_prefix']['sha256']},
        'No unsaved tail and no new journal lineage/copy at saved4000')
    require(record(S/'asr18_resident_serial350_DEV_v2.py')==base['resident_DEV_helper_source'],
        'Exact new full-processor helper source')
    return value


def checked_original_interval_resume(pin,*,base,folder,pointer_pin):
    """Exact actual CPU interval failure; never a normal0 or intentional stop."""
    import psutil
    from fastmss_owned_process_closure_v2 import dead
    require(pin==base['prior_resident_RAM_original_interval_failure'] and record(pin['path'])==pin,
        'Exact reviewed original-interval failure')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_resident_RAM_original_interval_failure_saved88400_tail88426_v1'
        and value['actual_owned_session']==62574 and value['actual_exec_tool_exit_code']==1
        and value['actual_wrapper_command_exit_code']==1
        and value['actual_native_OS_exit_code'] is None and value['actual_CPU_OS_exit_code'] is None
        and value['actual_native_nonzero_wait_observed_by_wrapper'] is True
        and value['normal_native0_claimed'] is False and value['intentional_stop_claimed'] is False
        and value['branch']==str(folder) and value['checkpoint_pointer']==pointer_pin
        and value['saved_optimizer_updates']==88400 and value['actual_optimizer_updates_before_failure']==88426
        and value['preserved_tail_positive_updates']==list(range(88401,88427))
        and all(dead(w)for w in value['captured_identity_pairs'])
        and all(not psutil.pid_exists(w['pid']) for w in value['known_old_PIDs_without_recorded_create_time']),
        'Actual failed owner dead, coherent88400 and 26 later positive attempts remain historical')
    for key in ('original_failure','original_monitor_completion','original_CPU_stderr','original_CPU_ready',
                'original_native_log','original_current_child','source_configuration','effective_configuration',
                'worker_source','ROOT_execution_review','checkpoint_pointer','checkpoint_persistence',
                'whole_journal_including_preserved_tail','preserved_unsaved_tail','actual_current_preparation_plan','unsubmitted_next_plan','failed_update_metadata'):
        require(record(value[key]['path'])==value[key],'Preserved actual failed source/state/tail evidence')
    current_plan=read_json(value['actual_current_preparation_plan']['path'])
    next_plan=read_json(value['unsubmitted_next_plan']['path'])
    remaining=read_json(value['failed_update_metadata']['path'])
    require(current_plan['initial_committed_updates']==88420 and len(current_plan['updates'])==20
        and next_plan['initial_committed_updates']==88440 and len(next_plan['updates'])==20
        and value['candidate_failed_global_updates']==list(range(88427,88441))
        and value['first_unconsumed_update_index']==6
        and remaining=={'source_plan':value['actual_current_preparation_plan'],'initial_updates':88420,
            'first_unconsumed_update_index':6,'candidate_global_updates':list(range(88427,88441)),
            'updates':current_plan['updates'][6:],'not_assumed_native_plus_one':True},
        'Actual current plan2042 and failed-before-enqueue plan2043; queued work is not falsely consumed')
    failure=read_json(value['original_failure']['path']);monitor=read_json(value['original_monitor_completion']['path'])
    require('Distinct ordered plan on healthy producer' in failure['error']
        and 'Original interval beyond source unresolved' in Path(value['original_CPU_stderr']['path']).read_text(encoding='utf-8')
        and failure['actual_optimizer_updates']==88426 and failure['last_checkpoint']==88400
        and monitor['joined'] is True and monitor['error'] is None
        and read_json(pointer_pin['path'])['journal_prefix']==value['saved_journal_prefix'],
        'Actual submit guard masked already failed ordinary CPU decoder; unchanged saved journal authority')
    return value


def private_successor_request(state,*,completed_boundaries,validation_stop,budget_stop):
    """Original MainThread pass bounds, plus the unchanged periodic stop."""
    if validation_stop or budget_stop:return None
    from asr18_BPE9223_lifecycle_v3 import next_boundary_request
    request=next_boundary_request(state,completed_boundaries)
    if request['kind']!='bounded_training_window':return None
    remaining=4000-request['initial_optimizer_updates']%4000
    return {'maximum_updates':min(request['maximum_updates'],remaining),
        'before_sampler_sha256':digest(state),'phase':request['phase']}


def adopt_private_successor(prefetch,ticket,schedule,submit_plan,*,wait,timings=None):
    """Only ordered full-row ingress changes; original planning stays MainThread."""
    if not wait and not prefetch.ready():
        if timings is not None:timings['not_ready_polls']+=1
        return None
    began=time.monotonic();prepared=prefetch.take(ticket,wait=wait);taken=time.monotonic()
    if timings is not None:
        timings.update(worker_preparation_seconds=prepared['preparation_seconds'],worker_metadata_bytes=prepared['metadata_bytes'],
            main_take_wait_seconds=taken-began,end_window_wait_count=int(wait))
    try:
        with prefetch.planning_rows(prepared,schedule):
            queued=submit_plan(schedule)
        require(queued[0]['initial_sampler_sha256']==digest(schedule.state_dict()),
            'Prepared successor begins at the exact private future state')
        if timings is not None:timings['main_adoption_and_submission_seconds']=time.monotonic()-taken
        return queued
    finally:del prepared


class Monitor:
    def __init__(self,folder):
        self.folder=folder;self.stop=threading.Event();self.error=None;self.samples=0;self.peak_private=0
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _observe(self):
        from resource_watchdog_v2 import sample_memory
        s=sample_memory(os.getpid(),'resident_RAM_system_headroom');self.samples+=1
        require(s['paired_sampling_within_one_second'],'Fresh actual process/system memory')
        self.peak_private=max(self.peak_private,s['worker_private_memory']['private_bytes'])
        m=s['system_memory']
        if m['commit_available_bytes']<2000000000 or m['physical_available_bytes']<1500000000:
            write_json(self.folder/'actual_failing_headroom_sample.json',s)
        require(m['commit_available_bytes']>=2000000000 and m['physical_available_bytes']>=1500000000,
            'Sensible real system headroom; no obsolete2GB CPU-family or90percent prospective ceiling')
        return s
    def _run(self):
        try:
            while not self.stop.wait(.5):self._observe()
        except BaseException as e:self.error=e
    def check(self):
        if self.error is not None:raise RuntimeError('Actual resident memory monitor: '+repr(self.error))
        require(not (self.folder/'stop.json').exists(),'Actual owner cooperative stop')
    def close(self):
        self.stop.set();self.thread.join(timeout=3);require(not self.thread.is_alive(),'Actual RAM monitor joined')
        write_json(self.folder/'monitor_completion.json',{'samples':self.samples,'maximum_sampled_private_bytes':self.peak_private,
            'joined':True,'clean':self.error is None,'error':None if self.error is None else repr(self.error)})


def main(a):
    import psutil
    from asr18_memory_diagnosis_v2 import storage
    storage();require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1','UTF8 before source reads')
    def E_reserve(prospective=0):
        require(shutil.disk_usage(ROOT).free>=100000000000+prospective,'E100GB reserve plus actual prospective output')
    E_reserve(64*1024**2)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','One explicitly visible training GPU')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    folder=Path(a.folder).resolve();require(folder.is_relative_to(ROOT/'runs'),'Exact E continuation branch')
    lease_path=P/'work/gpu_training_owner.json';lease=read_json(lease_path)
    require(lease['owner']==OWNER and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,
        'Exclusive idle training lease before resident GPU owner')
    producer={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':os.getppid()}
    base=read_json(a.configuration);pointer_pin=record(a.pointer);pointer=read_json(a.pointer)
    initial_updates=pointer['committed_updates']
    prefetch_policy=base['private_scalar_prefetch_policy']
    require(prefetch_policy=={'kind':'ASR18_private_scalar_row_prefetch_policy_v1',
        'source':record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
        'actual_CPU_control_review':record(ROOT/'analysis/BPE9223_training_launch_v1/ROOT_private_scalar_prefetch_actual_CPU_control_review_v1.json'),
        'additional_host_reservation_bytes':268435456,'maximum_TR_rows':280,'maximum_canonical_row_bytes':8388608,
        'SQLite_cache_bytes_each':4194304,'supported_phase':'weak_label_continued_training',
        'original_MainThread_planning_and_commits_unchanged':True,'idle_during_DEV':True},
        'Exact inactive-reviewed scalar prefetch source/resource policy; production still needs execution review')
    periodic_policy=base['periodic_DEV_policy']
    require(periodic_policy=={'kind':'ASR18_periodic_fixed350_DEV_monitor_policy_v1','interval_updates':4000,'first_update':4000,
        'scorer_policy':base['full_run_identity']['scorer_policy'],'periodic_result_path':str(folder/'periodic_DEV.json'),
        'full_pass_boundaries_and_winner_policy_unchanged':True},'Exact human-authorized periodic4000 fixed350 monitor policy')
    require(a.queue_bytes==268435456,'Changed bounded256MiB queue retains every full admissible256MiB packet')
    abnormal=None if a.abnormal_resume is None else checked_headroom_resume(record(a.abnormal_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    frontend_failure=None if a.DEV_frontend_resume is None else checked_DEV_frontend_resume(record(a.DEV_frontend_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    from asr18_resident_source_transition_v2 import checked_source_transition_resume
    transition=None if a.source_transition_resume is None else checked_source_transition_resume(
        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    interval_failure=None if a.original_interval_resume is None else checked_original_interval_resume(
        record(a.original_interval_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    require(sum(x is not None for x in (abnormal,frontend_failure,transition,interval_failure))<=1,'One truthful predecessor lane')
    preserved_tail=interval_failure if interval_failure is not None else(abnormal if abnormal is not None else(transition if transition is not None and transition['unsaved_tail_bytes'] else None))
    require(initial_updates>=260 and pointer['all_window_updates_committed'] is True
        and pointer['diagnostic_only'] is False
        and pointer['identity']['caller_identity']['full_run_lifecycle_identity']==base['full_run_identity'],
        'Exact coherent full-state source, no reset or diagnostic adoption')
    failed=read_json(a.latest_failure)
    require(failed['kind']=='ASR18_actual_ACKed260_singlepass_control_namespace_failure_TOOL1_v1'
        and failed['actual_tool_exit_code']==1 and failed['new_training_GPU_children']==failed['new_CPU_children']==0
        and failed['durable_ACK_committed'] is True,
        'Real last failed owner before any new CPU/GPU update; preserve genuine ACK260')
    from fastmss_owned_process_closure_v2 import dead
    require(dead(failed['producer']),'Actual previous parent dead')
    if not folder.exists():
        require(initial_updates==260 and failed['checkpoint_pointer']==pointer_pin,'Fresh branch only from genuine260')
        folder.mkdir();(folder/'checkpoints').mkdir();(folder/'progress').mkdir();(folder/'invocations').mkdir()
        write_json(folder/'owner.json',{'kind':'ASR18_exact_saved260_resident_RAM_branch_v1','source_checkpoint':pointer_pin,
            'identity':base['full_run_identity'],'original_RUN07_failure':record(a.latest_failure),'source':record(__file__)})
    else:
        owner=read_json(folder/'owner.json')
        require(owner['kind']=='ASR18_exact_saved260_resident_RAM_branch_v1' and owner['identity']==base['full_run_identity']
            and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
            and (a.prior_process is not None or abnormal is not None or frontend_failure is not None or transition is not None or interval_failure is not None),
            'Existing branch resumes its exact coherent RAM checkpoint and original scientific identity')
        if abnormal is None and frontend_failure is None and transition is None and interval_failure is None:
            prior=read_json(a.prior_process);report=read_json(prior['worker_report']['path'])
            require(record(prior['worker_report']['path'])==prior['worker_report'] and prior['actual_Popen_wait_code']==0
                and dead(prior['producer']) and report['latest_checkpoint']['pointer']==pointer_pin
                and report['all_live_updates_saved'] is True and report['actual_optimizer_updates']==initial_updates,
                'Genuine previous native normal wait0/death and exact latest all-state checkpoint')
        elif abnormal is not None:require(a.prior_process is None and a.invocation==4,'Resource failure is not a normal0 predecessor')
        elif frontend_failure is not None:require(a.prior_process is None and a.invocation==5,'Frontend failure is not a normal0 predecessor')
        elif interval_failure is not None:require(a.prior_process is None and a.invocation==10,'Actual original-interval TOOL1 is not normal0/intentional stop')
        else:require(a.prior_process is None and a.invocation>5,'Intentional TOOL1 source transition is never normal0')
    invocation=folder/'invocations'/('owner_'+str(a.invocation).zfill(6))
    require(not invocation.exists(),'Unique actual native invocation; preserve prior controls');invocation.mkdir();(invocation/'plans').mkdir()
    source_journal=pointer['journal_prefix']
    if abnormal is None and transition is None and interval_failure is None:
        require(record(source_journal['path'])=={'path':source_journal['path'],'bytes':source_journal['offset'],
            'sha256':source_journal['sha256']},'Exact whole coherent saved journal')
    else:
        require(source_journal==(interval_failure if interval_failure is not None else(abnormal if abnormal is not None else transition))['saved_journal_prefix'],'Exact reviewed saved prefix')
        SavedJournalPrefixView(source_journal,lambda:None)
    c=copy.deepcopy(base);c.update(worker_driver_source=record(__file__),parent_driver_source=record(__file__),
        resident_RAM_stream_configuration=record(a.stream_configuration),resident_branch_namespace=str(folder),
        checkpoint_interval_updates=200,source_checkpoint=pointer_pin,source_configuration=record(a.configuration))
    config_path=invocation/'configuration.json';write_json(config_path,c);c['_path']=str(config_path)
    write_json(invocation/'boot.json',{'producer':producer,'source':record(__file__),'configuration':record(config_path),
        'source_checkpoint':pointer_pin,'latest_failure':record(a.latest_failure),'old_RUN07_preserved':True})
    lease.update(running=True,actual_GPU_running=True,actual_training_optimizer_running=False,actual_owned_worker_pid=os.getpid(),
        actual_parent=producer,status='production_resident_RAM_restoring260',current_run_folder=str(folder),mode='training',
        current_configuration=record(config_path),committed_updates=initial_updates)
    write_json(lease_path,lease)
    monitor=Monitor(invocation);stream=None;engine=None;torch=None;began=time.monotonic();stage_stats={};scopes=ExitStack()
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
    try:
        require('torch' not in sys.modules,'Coherent compiler/linker preload beforeTorch')
        from preload_nemo_linker_v1 import preload
        linker=preload();require(linker['actual_binding_linker']['version']==[12,9] and not linker['cuda_initialized'],'Same CUDA compiler/runtime')
        import torch
        torch.set_num_threads(4)
        physical=torch.cuda.get_device_properties(0).total_memory
        require(physical>=17000000000,'Existing physical GPU capacity')
        torch.cuda.set_per_process_memory_fraction(12000000000/physical,device=0)
        def guard():monitor.check()
        def stage(name,**extra):
            guard();free,total=torch.cuda.mem_get_info()
            require(free-int(extra.get('prospective_joint_GPU_bytes',0))>=2000000000,'Physical desktop/GPU reserve')
            require(torch.cuda.memory_allocated()<=12000000000,'Same12GB CUDA allocator policy')
            stage_stats[name]=stage_stats.get(name,0)+1
            return {'cuda':{'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}}
        def admit(name,increment):
            guard();s=monitor._observe()
            reserve=prefetch_policy['additional_host_reservation_bytes']
            extra=0 if name=='private_scalar_prefetch_additional_host_state' else reserve
            require(s['system_memory']['commit_available_bytes']-increment-extra>=2000000000,
                'Actual allocation increment plus full extra prefetch reservation and unchanged commit headroom')
            require(s['system_memory']['physical_available_bytes']-reserve>=1500000000,
                'Full extra256MiB host reservation plus unchanged real physical headroom; no freed-memory credit')
            return {'passed':True,'increment':increment,'additional_prefetch_reservation_bytes':extra,'sample':s}
        from asr18_BPE9223_operational_metadata_v2 import configure
        @contextmanager
        def metadata_phase(name):guard();yield;guard()
        configure(**base['process_readonly_metadata_authority'],guard=guard,metadata_phase=metadata_phase)
        from native_tdt_minimal_import_v1 import prepare
        minimal=prepare()
        from asr18_token_only_carrier_v1 import load as token_carrier
        from stream_frozen_encoder_matrices_v1 import load
        from streamed_generation_device_v1 import apply
        base_path=Path((P/'work/base_model_path.txt').read_text().strip())
        model,loader=load(base_path);logical=apply(model);processor=token_carrier(base_path)
        teacher={name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False)for name in ('encoder_projector','decoder','joint')}
        from asr18_BPE9223_bootstrap_v2 import bootstrap
        optimizer,provenance,snapshot=bootstrap(model,processor,torch=torch,stage=stage,admit=admit,guard=guard,migration_plan=c['row_migration_plan'])
        from tdt_joint_canonical_2d_fp32_9223_v1 import apply as canonical_joint
        provenance.update(frozen_loader=record(S/'stream_frozen_encoder_matrices_v1.py'),bootstrap=record(S/'asr18_BPE9223_bootstrap_v2.py'),
            minimal_native=minimal,logical_device=logical,NEW_joint_numerical_policy=canonical_joint(model.joint),frozen_loader_policy=loader)
        from bpe_bounded_production_journal_v4 import JournalNamespace
        source_namespace=JournalNamespace(Path(source_journal['path']).parent,identity=c['full_run_identity'],guard=guard)
        def journal_factory(saved):
            if interval_failure is not None:
                require(saved==source_journal,'Exact actual88400 saved prefix, preserved26-tail never replayed')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved88400_prefix_after_original_interval_failure_v1',
                    'failure':record(a.original_interval_resume),'original_tail_modified':False,'no_tail_replay':True}
            if transition is not None:
                require(saved==source_journal,'Exact intentionally closed owner saved prefix')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_source_transition_saved_prefix_restore_v1',
                    'actual_transition':record(a.source_transition_resume),'no_tail_replay':True,'original_modified':False}
            if frontend_failure is not None:
                require(saved==source_journal and record(saved['path'])==frontend_failure['whole_journal'],'Exact whole saved4000 journal')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved4000_prefix_restore_after_DEV_frontend_failure_v1',
                    'failure':record(a.DEV_frontend_resume),'original_modified':False,'new_journal_branch_created':False}
            if abnormal is not None:
                require(saved==source_journal and record(saved['path'])==abnormal['whole_journal_including_preserved_tail'],'Exact preserved failed journal')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved3060_prefix_restore_with_retained_positive_tail_v1',
                    'failure':record(a.abnormal_resume),'original_tail_modified':False}
            require(saved==source_journal and record(saved['path'])=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']},
                'Saved whole coherent journal remains untouched during exact restore')
            return source_namespace.writer(Path(saved['path'])),{'kind':'ASR18_exact_coherent_readonly_journal_restore_v1','original_modified':False}
        from asr18_BPE9223_full_run_native_v12 import FullRunNativeOwner
        engine=FullRunNativeOwner(model,processor,teacher,optimizer,provenance,snapshot,invocation,c,torch=torch,stage=stage,
            admit=admit,guard=guard,journal=(SavedJournalPrefixView(source_journal,guard)if abnormal is not None or frontend_failure is not None or transition is not None or interval_failure is not None
                else source_namespace.writer(Path(source_journal['path']))),
            journal_factory=journal_factory,journal_factory_source=record(__file__))
        engine.restore_parent(pointer_pin,augmentation_identity=c['augmentation_identity'])
        require(optimizer.updates==initial_updates and not optimizer.poisoned and not engine.schedule.poisoned,'Actual854 restore before work')
        if preserved_tail is not None:
            destination=source_namespace.register('commits_'+str(a.invocation).zfill(6)+'.jsonl',prospective_bytes=source_journal['offset'])
            remaining=source_journal['offset']
            with Path(source_journal['path']).open('rb')as src,destination.open('ab')as dst:
                while remaining:
                    guard();block=src.read(min(1024**2,remaining));require(block,'Complete saved prefix remains available')
                    dst.write(block);remaining-=len(block)
                dst.flush();os.fsync(dst.fileno())
            require(record(source_journal['path'])==preserved_tail['whole_journal_including_preserved_tail'],'Old tail byteexact after prefix copy')
            journal=source_namespace.writer(destination)
            write_json(invocation/'actual_journal_branch.json',{'kind':'ASR18_actual_saved_prefix_recovery_journal_branch_v1',
                'predecessor':record(a.original_interval_resume if interval_failure is not None else(a.abnormal_resume if abnormal is not None else a.source_transition_resume)),
                'saved_prefix':source_journal,'preserved_original':preserved_tail['whole_journal_including_preserved_tail'],
                'new_active_journal':journal.binding(),'historical_unsaved_tail_updates':preserved_tail['preserved_tail_positive_updates'],
                'native_updates_or_draws_in_copy':0})
        elif initial_updates==260:
            namespace=JournalNamespace(folder/'journals',identity=c['full_run_identity'],guard=guard,create=True)
            destination=namespace.register('commits_000000.jsonl',prospective_bytes=source_journal['offset'])
            with Path(source_journal['path']).open('rb')as src,destination.open('ab')as dst:
                while True:
                    block=src.read(1024**2)
                    if not block:break
                    guard();dst.write(block)
                dst.flush();os.fsync(dst.fileno())
            journal=namespace.writer(destination)
        else:journal=source_namespace.writer(Path(source_journal['path']))
        require(journal.binding()['offset']==source_journal['offset']
            and journal.binding()['sha256']==source_journal['sha256'],'One exact original journal prefix copy, no tail replay')
        engine.journal=journal
        # Same already-proven runtime27 bank/backward/resident-GC policy.
        # The RAM transport changes neither this policy nor numerical updates.
        from asr18_frozen_encoder_cuda_cache_v3 import install,assert_idle
        from asr18_live_state_fingerprint_v1 import live_state_fingerprint
        from run_ASR18_native_throughput_component_v15 import cache_backward_calls,resident_gc_scope
        def owner_state_digest():return digest(live_state_fingerprint(optimizer,torch=torch,guard=guard))
        def admit_cache(request):
            require(request['logical_new_persistent_cuda_bytes']==1_217_066_496 and request['host_private_savings_credited_bytes']==0,'Entire frozenbank/no credit')
            admit('persistent294_cache_upload',1_217_066_496+4*1024**2)
            require(torch.cuda.memory_allocated()+1_217_066_496+16*1024**2<3_000_000_000,'Bank included in3GB endcache')
            return True
        policy=install(model,torch=torch,bank_binding=loader['bank'],guard=guard,admit_cache=admit_cache,owner_state_digest=owner_state_digest)
        write_json(invocation/'actual_cache_install.json',{'policy':policy,'idle':assert_idle(model,torch=torch,guard=guard)})
        scopes.enter_context(cache_backward_calls(model,torch=torch,enabled=True))
        scopes.enter_context(resident_gc_scope(invocation,enabled=True,optimizer=optimizer,torch=torch,guard=guard))
        from fastmss_resident_feature_stream_v5 import ResidentFeatureProducer
        resident_source=record(inspect.getsourcefile(ResidentFeatureProducer))
        from asr18_BPE9223_grouped_RAM_native_v1 import native_update
        from asr18_BPE9223_lifecycle_v3 import next_boundary_request,validate_window_plan
        from bpe_production_window_coordinator_v4 import WindowCoordinator
        from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
        lifecycle=read_json(folder/'lifecycle.json')if(folder/'lifecycle.json').exists()else {'completed_boundaries':[],'winner':base['qualified_source_winner']}
        periodic_path=folder/'periodic_DEV.json'
        periodic=read_json(periodic_path)if periodic_path.exists()else{'kind':'ASR18_periodic_fixed350_DEV_state_v1',
            'interval_updates':4000,'first_update':4000,'completed_evaluations':[]}
        require(periodic['kind']=='ASR18_periodic_fixed350_DEV_state_v1' and periodic['interval_updates']==4000
            and periodic['first_update']==4000,'Separate actual periodic monitor state')
        evaluated={v['optimizer_updates']for v in periodic['completed_evaluations']}
        require(len(evaluated)==len(periodic['completed_evaluations']) and all(type(v)is int and v>=4000 and v%4000==0 for v in evaluated),
            'Unique exact periodic evaluation frontiers')
        completed=len(lifecycle['completed_boundaries']);plan_sequence=0;queued=None;last_checkpoint=initial_updates
        latest=read_json(folder/'latest_checkpoint.json')
        require(latest['pointer']==pointer_pin and latest['full_state_verified'] is True,'Current coherent checkpoint before pending DEV')
        checkpoint_paths=sorted((folder/'checkpoints').glob('update_*'))[-5:];prepared_archive=[];winner_folders=set()
        if lifecycle['winner'].get('resident_checkpoint_folder'):winner_folders.add(lifecycle['winner']['resident_checkpoint_folder'])
        audio_seconds=0.;native_seconds=0.;active_began=time.monotonic();normal_reason=None
        lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
        restored_status={'kind':'ASR18_actual_resident_RAM_training_progress_v1','producer':producer,'source':record(__file__),
            'actual_optimizer_updates':initial_updates,'native_updates_this_process':0,'clocks':[x+initial_updates for x in (64250,2000,62250)],
            'committed_sampler_sha256':digest(engine.schedule.state_dict()),'journal_prefix':journal.binding(),'last_saved_updates':initial_updates,
            'native_seconds':0.,'active_pipeline_seconds':0.,'source_audio_seconds':0.,'audio_minutes_per_wall_minute':0.,
            'queue_bytes':a.queue_bytes,'optimizer_poisoned':False,'schedule_poisoned':False,'checkpoint_interval_updates':200,
            'planned_native_updates':401072,'administrative_update_stop':None,'phase':'coherent_restore_before_first_native',
            'periodic_DEV_interval_updates':4000}
        write_json(folder/'progress/latest.json',restored_status)
        write_json(folder/'active_journal.json',{'kind':'ASR18_actual_current_resident_training_journal_v1','journal_prefix':journal.binding(),
            'source_checkpoint':pointer_pin,'actual_optimizer_updates':initial_updates,'producer':producer,'source':record(__file__)})
        def submit_plan(schedule):
            nonlocal plan_sequence
            E_reserve(8*1024**2)
            remaining=4000-schedule.committed_updates()%4000
            plan=(engine.context.plan(schedule,completed_boundaries=completed)if remaining>=20 else
                engine.context.validation_plan(schedule,completed_boundaries=completed,maximum_updates=remaining));plan_sequence+=1
            path=invocation/'plans'/('plan_'+str(plan_sequence).zfill(6)+'.json');write_json(path,plan)
            pin=record(path);stream.submit(pin);return plan,pin
        def evaluate_coherent_DEV(destination):
            from asr18_resident_serial350_DEV_v2 import evaluate
            from asr18_BPE9223_source_context_v2 import full_run_next_draw_packet
            from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
            E_reserve(64*1024**2)
            before_schedule=copy.deepcopy(engine.schedule.state_dict());before_journal=journal.binding()
            before_next=full_run_next_draw_packet(engine.schedule,engine.context,optimizer_updates=optimizer.updates,guard=guard)
            before_lifecycle=copy.deepcopy(lifecycle);pending=None if queued is None else copy.deepcopy(queued[1])
            admission=admit('resident_optimizer_idle_serial350_DEV',2100000000)
            before_CUDA=stage('resident_serial350_DEV_before');started=time.monotonic();phase=PhasePrivateMonitor();phase.start()
            def DEV_guard():stage('resident_serial350_DEV_live')
            try:dev_pin,score=evaluate(model,processor,optimizer,torch=torch,config=c,pointer=latest['pointer'],folder=destination,guard=DEV_guard)
            finally:observed=phase.finish()
            require(observed['private_peak_bytes']<=admission['sample']['worker_private_memory']['private_bytes']+2100000000,
                'Measured resident DEV covered by unchanged2.1GB incremental allowance')
            require(engine.schedule.state_dict()==before_schedule and journal.binding()==before_journal and lifecycle==before_lifecycle,
                'DEV leaves both cursors/replay/journal/fullpass/winner state unchanged')
            require(full_run_next_draw_packet(engine.schedule,engine.context,optimizer_updates=optimizer.updates,guard=guard)==before_next,
                'DEV leaves actual next draw unchanged')
            if pending is not None:require(queued[1]==pending and record(pending['path'])==pending,'Pending private plan retained byteexact')
            resource={'kind':'ASR18_resident_DEV_actual_resource_and_frontier_preservation_v1','optimizer_updates':optimizer.updates,
                'DEV':dev_pin,'actual_native_updates_in_DEV':0,'actual_sampler_or_replay_advance':False,'all207_threeAdam_all4RNG_unchanged':True,
                'both_cursors_journal_and_nextdraw_unchanged':True,'fullpass_and_winner_state_unchanged_here':True,
                'pending_private_plan_retained':pending,'actual_phase_private':observed,'incremental_private_allowance_bytes':2100000000,
                'admission':admission,'CUDA_before':before_CUDA,'CUDA_after':stage('resident_serial350_DEV_after'),
                'CUDA_lifetime_peak_allocated_bytes_after':torch.cuda.max_memory_allocated(),'wall_seconds':time.monotonic()-started,
                'separate_concurrent_GPU_job':False,'resident_optimizer_objects_idle':True}
            path=Path(destination)/'training_frontier_preservation.json';write_json(path,resource)
            return dev_pin,score,record(path)
        def record_periodic(dev_pin,score,preservation,shared_fullpass):
            periodic['completed_evaluations'].append({'optimizer_updates':optimizer.updates,'DEV':dev_pin,'score':copy.deepcopy(score),
                'checkpoint_pointer':latest['pointer'],'frontier_preservation':preservation,'shared_with_full_pass_DEV':shared_fullpass,
                'periodic_record_advances_full_pass_or_winner':False})
            evaluated.add(optimizer.updates);write_json(periodic_path,periodic)
        if frontend_failure is not None:
            require(optimizer.updates==4000 and 4000 not in evaluated and queued is None,'Pending failed4000 evaluation before any4001/planning')
            lease.update(status='production_resident_RAM_serial350_DEV',actual_training_optimizer_running=False);write_json(lease_path,lease)
            restored_status.update(phase='pending_periodic_DEV_4000_before4001');write_json(folder/'progress/latest.json',restored_status)
            periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
            dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_000004000_attempt_'+str(a.invocation).zfill(6)))
            record_periodic(dev_pin,score,preservation,False)
            require(optimizer.updates==4000 and journal.binding()==source_journal,'Evaluation completed before positive Adam/new journal bytes')
            lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)
        plan,plan_pin=submit_plan(engine.schedule)
        from asr18_BPE9223_private_scalar_prefetch_v1 import PrivateScalarPrefetch
        if engine.schedule.phase=='weak_label_continued_training':
            prefetch=PrivateScalarPrefetch(engine.context,guard=guard,admit=admit)
            write_json(invocation/'private_metadata_worker_admission.json',prefetch.admission)
        while optimizer.updates<c['planned_native_updates']:
            action=next_boundary_request(engine.schedule.state_dict(),completed)
            if action['kind']!='bounded_training_window':
                raise RuntimeError('Declared complete350 DEV barrier reached; no pass/winner advance until real evaluator callback')
            validate_window_plan(engine.schedule.state_dict(),action,plan)
            window_started=time.monotonic();window_native_before=native_seconds;window_audio_before=audio_seconds
            metadata_timing={'not_ready_polls':0,'end_window_wait_count':0,'serial_fallback_count':0}
            private=engine.context.new_schedule();private.load_state_dict(normalize_schedule_state(plan['final_private_planning_sampler']))
            final_updates=plan['initial_committed_updates']+len(plan['updates'])
            validation_stop=a.validation_stop is not None and final_updates==a.validation_stop
            budget_stop=time.monotonic()-began>=a.maximum_wall_seconds-180 and final_updates-last_checkpoint>=200
            successor=private_successor_request(private.state_dict(),completed_boundaries=completed,
                validation_stop=validation_stop,budget_stop=budget_stop)
            queued=None;future_ticket=None
            if successor is not None:
                if prefetch is not None and successor['phase']=='weak_label_continued_training':
                    # Recheck fresh combined headroom before each new bounded private request.
                    admit('private_scalar_prefetch_additional_host_state',prefetch_policy['additional_host_reservation_bytes'])
                    future_ticket=prefetch.submit(private.state_dict(),maximum_updates=successor['maximum_updates'])
                else:
                    planning_started=time.monotonic();queued=submit_plan(private)
                    metadata_timing.update(serial_fallback_count=1,serial_planning_and_submission_seconds=time.monotonic()-planning_started)
            for index,entry in enumerate(plan['updates']):
                guard();packet=stream.next_update(digest(plan),index);metadata=packet.get('metadata',packet['update'])
                require(packet['plan_sha256']==digest(plan) and metadata['update_index']==index
                    and all(metadata[k]==entry[k]for k in ('native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','batch_id')),
                    'Current actual transaction matches privately prepared RAM packet')
                tx=engine.schedule.propose(optimizer_updates=optimizer.updates,resolve_TR=engine.context.resolve_identity)
                require(tx.identity_digest==entry['native_transaction_sha256'] and digest(tx.before)==entry['before_sampler_sha256'],
                    'Original source schedule/replay selection before positive Adam')
                rows=entry['rows'];arrays=packet['arrays'];require(len(arrays)==len(rows),'Every full row present')
                source=packet['source'];require(source['protocol']==resident_source,'Exact admitted resident CPU source, checked once at startup')
                binding={'kind':'ASR18_verified_resident_RAM_preparation_v1','metadata_sha256':digest(metadata),
                    'producer_source_sha256':source['protocol']['sha256'],'token':packet['token']}
                start=time.monotonic()
                try:
                    native=native_update(model,teacher,optimizer,rows,entry['draws'],feature_for_row=lambda i:arrays[i],
                        update_binding=binding,prepared_update=metadata,torch=torch,population_mean_U=engine.schedule.means[tx.phase],
                        update=optimizer.updates+1,stage=stage,stop_check=guard,phase=tx.phase,feature_update_index=index,
                        reference_original_head=False)
                    require(native['augmentation']==metadata['prepared_payload'],'Exact consumed frozen augmentation payload')
                    evidence=dict(binding,kind='ASR18_resident_RAM_preparation_commit_evidence_v1',actual_positive_native_update=optimizer.updates)
                    state=commit_RAM_transaction(tx,optimizer,journal,native,evidence)
                    require(digest(state)==entry['after_sampler_sha256'],'Exact both cursor/replay post-update state')
                except BaseException:
                    optimizer.poisoned=engine.schedule.poisoned=True;raise
                native_seconds+=time.monotonic()-start;audio_seconds+=sum(float(r['duration'])for r in rows)
                prepared_archive.append({'binding':binding,'metadata':metadata,'native':native});stream.release(packet['token']);del packet,arrays
                if future_ticket is not None:
                    queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=False,timings=metadata_timing)
                    if queued is not None:future_ticket=None
                if optimizer.updates==initial_updates+1 or optimizer.updates%20==0:
                    elapsed=time.monotonic()-active_began
                    status={'kind':'ASR18_actual_resident_RAM_training_progress_v1','producer':producer,'source':record(__file__),
                        'actual_optimizer_updates':optimizer.updates,'native_updates_this_process':optimizer.updates-initial_updates,
                        'clocks':[x+optimizer.updates for x in (64250,2000,62250)],'committed_sampler_sha256':digest(state),
                        'journal_prefix':journal.binding(),'last_saved_updates':last_checkpoint,'native_seconds':native_seconds,
                        'active_pipeline_seconds':elapsed,'source_audio_seconds':audio_seconds,
                        'audio_minutes_per_wall_minute':audio_seconds/elapsed,'queue_bytes':a.queue_bytes,
                        'optimizer_poisoned':optimizer.poisoned,'schedule_poisoned':engine.schedule.poisoned,
                        'checkpoint_interval_updates':200,'planned_native_updates':401072,'administrative_update_stop':None}
                    write_json(folder/'progress/latest.json',status);print(json.dumps(status),flush=True)
            if future_ticket is not None:
                queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=True,timings=metadata_timing)
                future_ticket=None
            require(prefetch is None or(prefetch.outstanding is None and prefetch.held_result is None),
                'All ordered metadata adopted and submitted; no active/pending metadata during checkpoint or DEV')
            del private
            periodic_due=optimizer.updates>=4000 and optimizer.updates%4000==0 and optimizer.updates not in evaluated
            need_checkpoint=optimizer.updates-last_checkpoint>=200 or queued is None or periodic_due
            checkpoint_started=time.monotonic()
            if need_checkpoint:
                E_reserve(pointer['payload']['bytes']+64*1024**2)
                from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
                from asr18_streamed_persistence_fingerprint_v1 import verify_persistence
                from asr18_BPE9223_source_context_v2 import full_run_next_draw_packet
                from asr18_BPE9223_training_state_v1 import checkpoint
                ck=folder/'checkpoints'/('update_'+str(optimizer.updates).zfill(9));ck.mkdir()
                before=live_state_fingerprint(optimizer,torch=torch,guard=guard);write_json(ck/'pre_save.json',before)
                nxt=full_run_next_draw_packet(engine.schedule,engine.context,optimizer_updates=optimizer.updates,guard=guard);write_json(ck/'next_draw.json',nxt)
                with gzip.open(ck/'prepared_metadata.json.gz','xb')as f:f.write(canonical(prepared_archive))
                caller={'kind':'ASR18_resident_RAM_training_checkpoint_caller_v1','full_run_lifecycle_identity':c['full_run_identity'],
                    'run_identity_sha256':digest(c['full_run_identity']),'config':record(config_path),'engine':record(__file__),
                    'session':c['session'],'pre_save_fingerprint':record(ck/'pre_save.json'),'next_draw_packet':record(ck/'next_draw.json'),
                    'native_forecast_high_water':engine.forecast.state_dict(),'preparation_archive':record(ck/'prepared_metadata.json.gz'),
                    'checkpoint_span_updates':optimizer.updates-last_checkpoint,'checkpoint_interval_updates':200,
                    'geometry_forecast_used_for_hotpath_admission':False,
                    'source_checkpoint':pointer_pin,'native_source':record(S/'asr18_BPE9223_grouped_RAM_native_v1.py')}
                coordinator=WindowCoordinator.__new__(WindowCoordinator)
                coordinator.schedule=engine.schedule;coordinator.optimizer=optimizer;coordinator.journal=journal
                coordinator.plan=plan;coordinator.index=len(plan['updates']);coordinator.poisoned=False
                def writer(saved,identity):
                    return checkpoint(ck/'checkpoint.json',torch=torch,optimizer=optimizer,state=saved,provenance=provenance,
                        identity=identity,augmentation=engine.binding,journal=journal,diagnostic_only=False,guard=guard)
                ptr=coordinator.publish_checkpoint(write_coherent_checkpoint=writer,pointer_path=ck/'pointer.json',checkpoint_identity=caller,diagnostic_only=False)
                proof=verify_persistence(ptr['checkpoint'],expected_pre_save=before,identity=ptr['identity'],sampler_state=engine.schedule.state_dict(),
                    journal_prefix=journal.binding(),diagnostic_only=False,guard=guard)
                require(proof['named_tensors_verified']==854 and proof['Adam_states_verified']==215,'Full854/215 checkpoint persistence')
                require_exact_live_restore(before,live_state_fingerprint(optimizer,torch=torch,guard=guard))
                write_json(ck/'persistence_verified.json',proof);latest={'pointer':record(ck/'pointer.json'),'persistence':record(ck/'persistence_verified.json'),
                    'committed_updates':optimizer.updates,'full_state_verified':True,'source':record(__file__)}
                write_json(folder/'latest_checkpoint.json',latest);last_checkpoint=optimizer.updates;checkpoint_paths.append(ck);prepared_archive=[]
                # Exact branch-owned payload retirement only; preserve all metadata and original RUN07.
                for old in checkpoint_paths[:-5]:
                    if str(old)in winner_folders:continue
                    meta=read_json(old/'checkpoint.json');payload=Path(meta['payload']['path']).resolve()
                    require(payload.parent==old and payload.is_relative_to(folder/'checkpoints'),'Exact owned old branch payload only')
                    if payload.exists():require(record(payload)==meta['payload'],'Unchanged owned retired payload');payload.unlink()
                checkpoint_paths=checkpoint_paths[-5:]
                print(json.dumps({'actual_saved_updates':last_checkpoint,'checkpoint':latest,'old_RUN07_untouched':True}),flush=True)
            metadata_timing.update(kind='ASR18_actual_private_metadata_window_timing_v1',
                initial_updates=plan['initial_committed_updates'],committed_updates=optimizer.updates,
                native_seconds=native_seconds-window_native_before,source_audio_seconds=audio_seconds-window_audio_before,
                checkpoint_included=need_checkpoint,checkpoint_seconds=time.monotonic()-checkpoint_started if need_checkpoint else 0.,
                window_pipeline_seconds=time.monotonic()-window_started)
            with(invocation/'private_metadata_window_timings.jsonl').open('a',encoding='utf-8')as timing_log:
                timing_log.write(json.dumps(metadata_timing,separators=(',',':'))+'\n')
            action=next_boundary_request(engine.schedule.state_dict(),completed)
            if action['kind']=='checkpoint_then_serial350DEV':
                CPU_closed=stream.close();stream=None;write_json(invocation/('CPU_pass_'+str(completed)+'_normal0.json'),CPU_closed)
                dev_pin,score,preservation=evaluate_coherent_DEV(folder/('DEV_'+str(completed+1)))
                if periodic_due:record_periodic(dev_pin,score,preservation,True)
                old=lifecycle['winner']
                if score['strict_all_six_passed']and score['word_errors']['tr']<old['word_errors']['tr']:
                    lifecycle['winner']={'artifact':ptr['checkpoint'],'checkpoint_pointer':latest['pointer'],'word_errors':score['word_errors'],
                        'reference_words':score['reference_words'],'DEV':dev_pin,'strict_qualified':True,'resident_checkpoint_folder':str(ck)}
                    winner_folders.add(str(ck))
                lifecycle['completed_boundaries'].append({'boundary':action['boundary'],'DEV':dev_pin,'score':score,'checkpoint_pointer':latest['pointer']})
                completed+=1;write_json(folder/'lifecycle.json',lifecycle)
                if completed==3:normal_reason='all401072_and_three_declared_DEVs_complete';break
                if validation_stop or budget_stop:
                    normal_reason='first200_checkpoint_fresh_restore_validation'if validation_stop else 'normal_owner_wall_budget'
                    break
                stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/('CPU_after_DEV_'+str(completed)),queue_bytes=a.queue_bytes,guard=guard)
                plan,plan_pin=submit_plan(engine.schedule);continue
            if periodic_due:
                periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
                dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_'+str(optimizer.updates).zfill(9)))
                record_periodic(dev_pin,score,preservation,False)
            if validation_stop or budget_stop:
                normal_reason='first200_checkpoint_fresh_restore_validation'if validation_stop else 'normal_owner_wall_budget'
                break
            require(queued is not None,'Exact private successor after periodic or normal bounded window')
            plan,plan_pin=queued
        require(last_checkpoint==optimizer.updates,'Normal close saves every live update')
        close_private_metadata() # Genuine joined closure before normal0/engine context closure.
        if stream is not None:
            CPU_closed=stream.close();stream=None;write_json(invocation/'CPU_normal0.json',CPU_closed)
        latest=read_json(folder/'latest_checkpoint.json')
        write_json(invocation/'normal_completion.json',{'kind':'ASR18_actual_resident_RAM_coherent_native_completion_v1','passed':True,
            'producer':producer,'source':record(__file__),'actual_optimizer_updates':optimizer.updates,'all_live_updates_saved':True,
            'latest_checkpoint':latest,'reason':normal_reason,'completed_boundaries':completed,'actual_Popen_wait_still_unobserved_inside_process':True,
            'elapsed_seconds':time.monotonic()-began,'native_seconds':native_seconds,'source_audio_seconds':audio_seconds})
        if completed==3:write_json(folder/'training_complete.json',{'actual_optimizer_updates':optimizer.updates,'last_checkpoint':last_checkpoint,'completed_boundaries':completed})
    except BaseException as error:
        if stream is not None:
            try:stream.abort()
            except BaseException as abort_error:write_json(invocation/'CPU_abort_error.json',{'error':repr(abort_error),'normal_CPU0_claimed':False})
            stream=None
        write_json(invocation/'actual_failure.json',{'producer':producer,'error':repr(error),'traceback':traceback.format_exc(),
            'actual_optimizer_updates':None if engine is None else engine.optimizer.updates,'last_checkpoint':locals().get('last_checkpoint',initial_updates),
            'all_unsaved_live_updates_require_coherent_reload':True,'elapsed_seconds':time.monotonic()-began})
        raise
    finally:
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--stream-configuration',required=True)
    p.add_argument('--pointer',required=True);p.add_argument('--latest-failure',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--queue-bytes',type=int,default=268435456);p.add_argument('--invocation',type=int,default=1)
    p.add_argument('--original-interval-resume');p.add_argument('--source-transition-resume');p.add_argument('--prior-process');p.add_argument('--abnormal-resume');p.add_argument('--DEV-frontend-resume');p.add_argument('--validation-stop',type=int)
    p.add_argument('--maximum-wall-seconds',type=int,default=86400);main(p.parse_args())
