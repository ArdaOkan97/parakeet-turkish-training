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
            and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints') and a.prior_process is not None,
            'Existing branch resumes its exact coherent RAM checkpoint and original scientific identity')
        prior=read_json(a.prior_process);report=read_json(prior['worker_report']['path'])
        require(record(prior['worker_report']['path'])==prior['worker_report'] and prior['actual_Popen_wait_code']==0
            and dead(prior['producer']) and report['latest_checkpoint']['pointer']==pointer_pin
            and report['all_live_updates_saved'] is True and report['actual_optimizer_updates']==initial_updates,
            'Genuine previous native normal wait0/death and exact latest all-state checkpoint')
    invocation=folder/'invocations'/('owner_'+str(a.invocation).zfill(6))
    require(not invocation.exists(),'Unique actual native invocation; preserve prior controls');invocation.mkdir();(invocation/'plans').mkdir()
    source_journal=pointer['journal_prefix'];require(record(source_journal['path'])=={
        'path':source_journal['path'],'bytes':source_journal['offset'],'sha256':source_journal['sha256']},'Exact whole saved260 journal')
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
            guard();s=monitor._observe();require(s['system_memory']['commit_available_bytes']-increment>=2000000000,
                'Actual allocation increment plus real system headroom')
            return {'passed':True,'increment':increment,'sample':s}
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
            require(saved==source_journal and record(saved['path'])=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']},
                'Saved whole coherent journal remains untouched during exact restore')
            return source_namespace.writer(Path(saved['path'])),{'kind':'ASR18_exact_coherent_readonly_journal_restore_v1','original_modified':False}
        from asr18_BPE9223_full_run_native_v12 import FullRunNativeOwner
        engine=FullRunNativeOwner(model,processor,teacher,optimizer,provenance,snapshot,invocation,c,torch=torch,stage=stage,
            admit=admit,guard=guard,journal=source_namespace.writer(Path(source_journal['path'])),
            journal_factory=journal_factory,journal_factory_source=record(__file__))
        engine.restore_parent(pointer_pin,augmentation_identity=c['augmentation_identity'])
        require(optimizer.updates==initial_updates and not optimizer.poisoned and not engine.schedule.poisoned,'Actual854 restore before work')
        if initial_updates==260:
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
        from fastmss_resident_feature_stream_v1 import ResidentFeatureProducer
        resident_source=record(inspect.getsourcefile(ResidentFeatureProducer))
        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)
        from asr18_BPE9223_grouped_RAM_native_v1 import native_update
        from asr18_BPE9223_lifecycle_v3 import next_boundary_request,validate_window_plan
        from bpe_production_window_coordinator_v4 import WindowCoordinator
        from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
        lifecycle=read_json(folder/'lifecycle.json')if(folder/'lifecycle.json').exists()else {'completed_boundaries':[],'winner':base['qualified_source_winner']}
        completed=len(lifecycle['completed_boundaries']);plan_sequence=0;queued=None;last_checkpoint=initial_updates
        checkpoint_paths=sorted((folder/'checkpoints').glob('update_*'))[-5:];prepared_archive=[];winner_folders=set()
        if lifecycle['winner'].get('resident_checkpoint_folder'):winner_folders.add(lifecycle['winner']['resident_checkpoint_folder'])
        audio_seconds=0.;native_seconds=0.;active_began=time.monotonic();normal_reason=None
        lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
        def submit_plan(schedule):
            nonlocal plan_sequence
            E_reserve(8*1024**2)
            plan=engine.context.plan(schedule,completed_boundaries=completed);plan_sequence+=1
            path=invocation/'plans'/('plan_'+str(plan_sequence).zfill(6)+'.json');write_json(path,plan)
            pin=record(path);stream.submit(pin);return plan,pin
        plan,plan_pin=submit_plan(engine.schedule)
        while optimizer.updates<c['planned_native_updates']:
            action=next_boundary_request(engine.schedule.state_dict(),completed)
            if action['kind']!='bounded_training_window':
                raise RuntimeError('Declared complete350 DEV barrier reached; no pass/winner advance until real evaluator callback')
            validate_window_plan(engine.schedule.state_dict(),action,plan)
            private=engine.context.new_schedule();private.load_state_dict(normalize_schedule_state(plan['final_private_planning_sampler']))
            final_updates=plan['initial_committed_updates']+len(plan['updates'])
            validation_stop=a.validation_stop is not None and final_updates==a.validation_stop
            budget_stop=time.monotonic()-began>=a.maximum_wall_seconds-180 and final_updates-last_checkpoint>=200
            if not(validation_stop or budget_stop)and next_boundary_request(private.state_dict(),completed)['kind']=='bounded_training_window':queued=submit_plan(private)
            else:queued=None
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
                if optimizer.updates==261 or optimizer.updates%20==0:
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
            need_checkpoint=optimizer.updates-last_checkpoint>=200 or queued is None
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
            if validation_stop or budget_stop:
                normal_reason='first200_checkpoint_fresh_restore_validation'if validation_stop else 'normal_owner_wall_budget'
                break
            if queued is None:
                action=next_boundary_request(engine.schedule.state_dict(),completed)
                require(action['kind']=='checkpoint_then_serial350DEV','Only actual pass/DEV barrier')
                CPU_closed=stream.close();stream=None;write_json(invocation/('CPU_pass_'+str(completed)+'_normal0.json'),CPU_closed)
                from asr18_resident_serial350_DEV_v1 import evaluate
                dev_pin,score=evaluate(model,processor,optimizer,torch=torch,config=c,pointer=latest['pointer'],
                    folder=folder/('DEV_'+str(completed+1)),guard=guard)
                old=lifecycle['winner']
                if score['strict_all_six_passed']and score['word_errors']['tr']<old['word_errors']['tr']:
                    lifecycle['winner']={'artifact':ptr['checkpoint'],'checkpoint_pointer':latest['pointer'],'word_errors':score['word_errors'],
                        'reference_words':score['reference_words'],'DEV':dev_pin,'strict_qualified':True,'resident_checkpoint_folder':str(ck)}
                    winner_folders.add(str(ck))
                lifecycle['completed_boundaries'].append({'boundary':action['boundary'],'DEV':dev_pin,'score':score,'checkpoint_pointer':latest['pointer']})
                completed+=1;write_json(folder/'lifecycle.json',lifecycle)
                if completed==3:normal_reason='all401072_and_three_declared_DEVs_complete';break
                stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/('CPU_after_DEV_'+str(completed)),queue_bytes=a.queue_bytes,guard=guard)
                plan,plan_pin=submit_plan(engine.schedule);continue
            plan,plan_pin=queued
        require(last_checkpoint==optimizer.updates,'Normal close saves every live update')
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
        if stream is not None:stream.close()
        scopes.close()
        if engine is not None:engine.close()
        monitor.close()
        lease=read_json(lease_path)
        if lease.get('actual_owned_worker_pid')==os.getpid():
            lease.update(running=False,actual_GPU_running=False,actual_training_optimizer_running=False,actual_owned_worker_pid=None,
                status='production_resident_RAM_process_closed',committed_updates=locals().get('last_checkpoint',initial_updates));write_json(lease_path,lease)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--stream-configuration',required=True)
    p.add_argument('--pointer',required=True);p.add_argument('--latest-failure',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--queue-bytes',type=int,default=1073741824);p.add_argument('--invocation',type=int,default=1)
    p.add_argument('--prior-process');p.add_argument('--validation-stop',type=int)
    p.add_argument('--maximum-wall-seconds',type=int,default=86400);main(p.parse_args())
