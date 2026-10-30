"""Inactive resident full-run native owner: one bounded window in live memory.

Normal publication is one complete-window-end PRE/SAVED/persistence checkpoint.
This source never launches a process or qualifies a winner. Parent admission,
indexed namespace ownership, current fresh recovery, DEV and durable ack remain
separate exact contracts. Diagnostics are never accepted as restore seeds.
"""
from copy import deepcopy
import json,time
from pathlib import Path
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from bpe_training_schedule_v1 import WEAK,CLEANER
from asr18_BPE9223_lifecycle_v3 import progress,validate_window_plan,next_boundary_request
from bpe_production_window_coordinator_v4 import WindowCoordinator
from asr18_BPE9223_durable_owner_adapter_v3 import PackedDurableLifecycleAdapter
from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223,full_run_next_draw_packet
from asr18_BPE9223_training_state_v1 import checkpoint as save_state,restore as restore_state
from bpe_broad_production_v3 import assert_model,prohibit_model_transforms
from asr18_BPE9223_grouped_native_v1 import native_update
from asr18_BPE9223_metadata_checkpoint_binding_v2 import MetadataCheckpointBinding
from asr18_BPE9223_geometry_forecast_v1 import CalibratedGeometryForecast
from asr18_full_population_separated_forecast_v1 import phase_projection
from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
from asr18_streamed_persistence_fingerprint_v1 import verify_persistence,normalize_fingerprint
from fastmss_BPE9223_feature_protocol_v8 import FeatureWindowReader as PhaseAwareFeatureWindowReader
from asr18_BPE9223_operational_metadata_v2 import read_bound
from resource_watchdog_v2 import sample_memory


def verify_merged_CPU_native_handoff(command,config,*,optimizer_updates,sampler_sha256,guard):
    from run_ASR18_BPE9223_CPU_window_v10 import checked_task_completion
    from fastmss_owned_process_closure_v2 import dead
    task_pin=command['CPU_task_completion'];report_pin=command['CPU_task_report']
    task=read_bound(task_pin,guard);report=read_bound(report_pin,guard)
    checked_task_completion(task_pin,report_pin,task['configuration'],config['operational_compatibility'],guard=guard)
    witness=read_bound(command['CPU_task_handoff_witness'],guard)
    semantic={key:value for key,value in command.items() if key!='CPU_task_handoff_witness'}
    require(witness['kind']=='ASR18_fresh_live_parent_no_CPU_inflight_native_task_handoff_v1'
        and witness['source']==config['full_owner_controller_source'] and witness['parent_source']==config['parent_driver_source']
        and witness['producer']==task['producer'] and not dead(witness['producer'])
        and witness['task_nonce']==task['task_nonce'] and witness['CPU_task_completion']==task_pin
        and witness['packet_semantic_sha256']==digest(semantic)
        and witness['actual_CPU_leaf_inflight'] is False and witness['task_family_observer_joined'] is True
        and 0<=time.monotonic()-witness['observed_monotonic']<=2
        and witness['optimizer_updates']==optimizer_updates and witness['sampler_sha256']==sampler_sha256
        and task['source']==config['CPU_controller_source']
        and task['owner_binding']==command['feature_owner'] and task['manifest']==command['feature_manifest']
        and report['window_index']==command['window_index'] and report['bridge']==read_bound(report['source_contract'],guard)['bridge'],
        'Fresh live actual parent/task/whole-array/orderedpayload source authority before first native draw')
    return True


class FullRunNativeOwner:
    def __init__(self,model,processor,teacher,optimizer,provenance,bootstrap_snapshot,folder,config,
                 *,torch,stage,admit,guard,journal,journal_factory,journal_factory_source):
        require(config['kind']=='ASR18_full_run_native_owner_configuration_v1'
                and config['production_enabled'] is False and type(config['diagnostic_only']) is bool and config['diagnostic_only']==config['owner_validation_only']
                and config['planned_native_updates']==401072,'Exact inactive full-run preparation identity')
        from asr18_temporal_tiled_subsampling_v2 import apply
        require(record(Path(__file__).with_name('asr18_temporal_tiled_subsampling_v2.py'))==config['validated_conv_only_source'],
                'Exact validated conv-only component')
        provenance['validated_conv_only_CNN_lifetime_policy']=apply(model,torch=torch,guard=guard)
        self.model=model;self.teacher=teacher;self.optimizer=optimizer;self.torch=torch
        self.provenance=provenance;self.config=config;self.folder=Path(folder)
        self.stage=stage;self.admit=admit;self.guard=guard
        self.context=FullRunSourceContext9223(config['planning_arguments'],guard=guard)
        require(digest(bootstrap_snapshot)==digest(self.context.snapshot) and optimizer.updates==0,
                'Original BPE64250/LoRA62250 mature207/3Adam/RNG bootstrap only')
        self.schedule=self.context.new_schedule();self.forecast=CalibratedGeometryForecast(**config['forecast_calibration'],guard=guard)
        self.journal=journal;self.journal_factory=journal_factory;self.journal_factory_source=journal_factory_source
        self.process_initial_updates=0;self.native_in_process=0;self.window_count=0
        self.binding=None;self.pointer=None;self.capture=None;self.coordinator=None;self.reader=None;self.inflight=False
        self.active_folder=self.folder;self.phase_records=[];self.last_window=None
        self.identity=deepcopy(config['full_run_identity']);self.last_persistence=None
        self.checkpoint_writer_open=False;self.checkpoint_reader_open=False;self.checkpoint_restore_open=False;self.candidate_intent=None
        prohibit_model_transforms(model,torch);model._broad_processor=processor

    def _private(self,label):
        sample=sample_memory(__import__('os').getpid(),label)
        require(sample['paired_sampling_within_one_second'] and sample['system_memory']['available']
                and sample['worker_private_memory']['available'],'Fresh paired native observation')
        return sample['worker_private_memory']['private_bytes']

    def _progress(self,*,native=None,committed=False,error=None):
        write_json(self.active_folder/'native_progress.json',{'kind':'ASR18_full_run_actual_native_progress_v1',
            'actual_optimizer_updates':self.optimizer.updates,'native_updates_this_process':self.optimizer.updates-self.process_initial_updates,
            'native_return_complete':native is not None,'journal_sampler_committed':committed,
            'clocks':[n+self.optimizer.updates for n in (64250,2000,62250)],
            'optimizer_poisoned':self.optimizer.poisoned,'schedule_poisoned':self.schedule.poisoned,
            'run_identity_sha256':digest(self.identity),'error':error,'production_enabled':False})

    def _run_phase(self,name,*,phase,increment,operation):
        self.guard();before=self._private(name+'_before')
        projection=phase_projection(self.config['entire_worker_forecast'],phase=phase,current_private_bytes=before,increment_bytes=increment)
        self.admit(name,increment);monitor=PhasePrivateMonitor();monitor.start();began=time.monotonic();returned=False
        try:result=operation();returned=True
        finally:
            proof=monitor.finish();proof['wall_seconds']=time.monotonic()-began;covered=proof['private_peak_bytes']<=before+increment
            path=self.active_folder/(name+'_memory.json')
            write_json(path,{'before_private_bytes':before,'increment_bytes':increment,'projection':projection,
                'actual':proof,'covered':covered,'margin_bytes':before+increment-proof['private_peak_bytes']})
            require(len(self.phase_records)<24,'Only current window phase observations in live state')
            self.phase_records.append({'name':name,'phase':phase,'memory_record':record(path),
                'wall_seconds':proof['wall_seconds'],'operation_returned':returned,'forecast_covered':covered})
        require(covered,'Prospective phase missed: preserve actual frontier, never unchanged retry')
        return result,proof

    def restore_parent(self,pointer_binding,*,augmentation_identity):
        pointer=read_bound(pointer_binding,self.guard)
        require(pointer['diagnostic_only']==self.config['diagnostic_only'] and pointer['all_window_updates_committed'] is True
                and pointer['identity']['caller_identity']['full_run_lifecycle_identity']==self.identity,
                'No diagnostic seed or different full-run identity')
        self.binding=MetadataCheckpointBinding(augmentation_identity,guard=self.guard)
        def operation():
            self.checkpoint_restore_open=True
            try:
                state,journal,branch=restore_state(pointer['checkpoint'],torch=self.torch,model=self.model,optimizer=self.optimizer,
                    identity=pointer['identity'],augmentation=self.binding,journal_factory=self.journal_factory,
                    journal_factory_source=self.journal_factory_source,diagnostic_only=self.config['diagnostic_only'],guard=self.guard)
            finally:self.checkpoint_restore_open=False
            from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
            self.schedule.load_state_dict(normalize_schedule_state(state));self.journal=journal
            caller=pointer['identity']['caller_identity']
            expected=normalize_fingerprint(read_bound(caller['pre_save_fingerprint'],self.guard))
            require_exact_live_restore(expected,live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard))
            packet=full_run_next_draw_packet(self.schedule,self.context,optimizer_updates=self.optimizer.updates,guard=self.guard)
            require(digest(packet)==digest(read_bound(caller['next_draw_packet'],self.guard)),'Actual source next draw restored exactly')
            self.forecast.load_state_dict(caller['native_forecast_high_water'])
            return state,branch
        (state,branch),memory=self._run_phase('startup_current_restore',phase='startup_restore',increment=350000000,operation=operation)
        require(self.optimizer.updates==progress(state)['optimizer_updates']==pointer['committed_updates']
                and digest(state)==pointer['sampler_state_sha256'],'Actual full-run mature/source frontier restored')
        self.process_initial_updates=self.optimizer.updates;self.pointer=deepcopy(pointer_binding)
        write_json(self.folder/'actual_parent_restore.json',{'passed':True,'pointer':pointer_binding,'branch':branch,
            'memory':memory,'all207_threeAdam_all4RNG_match_actual_pre_save':True,'next_draw_packet_match':True,
            'restore_occurred_only_at_startup_before_native':True,'production_enabled':False})

    def _checkpoint(self,command):
        state=self.schedule.state_dict();update=self.optimizer.updates;folder=self.active_folder;intent=command['checkpoint_intent']
        require(intent['publication_role']=='completed_window_end','One reserved normal window-end publication')
        caller={'kind':'ASR18_resident_full_run_persistence_caller_identity_v1','full_run_lifecycle_identity':self.identity,'run_identity_sha256':digest(self.identity),
            'BPE9223_population_sha256':self.config['planning_arguments']['DATA_evidence']['BPE9223_population']['sha256'],
            'lifecycle_parent_head':command['lifecycle_parent_head'],'session':self.config['session'],'config':record(self.config['_path']),
            'engine':record(__file__),'native':record(Path(__file__).with_name('asr18_BPE9223_grouped_native_v1.py')),
            'feature_window':command['feature_manifest'],'feature_owner':command['feature_owner'],
            'native_forecast_high_water':self.forecast.state_dict(),'production_enabled':False,
            'exact_GPU_restore_deferred_until_actual_native_exit':True}
        def operation():
            before=live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard)
            pre=folder/'actual_pre_save_fingerprint.json';nxt=folder/'actual_next_draw_packet.json'
            require(not pre.exists() and not nxt.exists(),'Fresh actual independent PRE/save evidence')
            write_json(pre,before);write_json(nxt,full_run_next_draw_packet(self.schedule,self.context,optimizer_updates=update,guard=self.guard))
            caller.update(pre_save_fingerprint=record(pre),next_draw_packet=record(nxt))
            def writer(saved,identity):
                require(digest(saved)==digest(state),'Actual pre-writer sampler')
                self.checkpoint_writer_open=True
                try:
                    return save_state(intent['metadata_path'],torch=self.torch,optimizer=self.optimizer,state=saved,
                        provenance=self.provenance,identity=identity,augmentation=self.binding,journal=self.journal,
                        diagnostic_only=self.config['diagnostic_only'],guard=self.guard)
                finally:self.checkpoint_writer_open=False
            pointer=self.coordinator.publish_checkpoint(write_coherent_checkpoint=writer,pointer_path=intent['pointer_path'],
                checkpoint_identity=caller,diagnostic_only=self.config['diagnostic_only'])
            self.pointer=record(intent['pointer_path'])
            self.checkpoint_reader_open=True
            try:
                proof=verify_persistence(pointer['checkpoint'],expected_pre_save=before,identity=pointer['identity'],sampler_state=state,
                    journal_prefix=pointer['journal_prefix'],diagnostic_only=self.config['diagnostic_only'],guard=self.guard)
            finally:self.checkpoint_reader_open=False
            require(proof['named_tensors_verified']==854 and proof['Adam_states_verified']==215,'Exact9223 payload854/all215 state views')
            after=live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard);require_exact_live_restore(before,after)
            require(digest(self.schedule.state_dict())==digest(state) and self.journal.binding()==pointer['journal_prefix'],
                    'Read-only persistence never mutates live science')
            proof.update(checkpoint_pointer=self.pointer,pre_save_fingerprint=record(pre),next_draw_packet=record(nxt),
                live_after_state_sha256=after['state_sha256'],live_unchanged_after_persistence=True,
                worker_source=record(self.config['worker_driver_source']['path']),engine_source=record(__file__),actual_GPU_restore_performed=False,
                deferred_restore_required_before_overall_diagnostic_pass=True)
            path=folder/'persistence_verified.json';write_json(path,proof);self.last_persistence=record(path)
            return pointer,proof
        (pointer,proof),memory=self._run_phase('window_end_'+str(update),phase='checkpoint_write',increment=64000000,operation=operation)
        self.capture={'pointer':self.pointer,'checkpoint':pointer['checkpoint'],'write_and_persistence_memory':memory,
            'checkpoint_persistence_proof':self.last_persistence,'pre_save_fingerprint':proof['pre_save_fingerprint'],
            'next_draw_packet':proof['next_draw_packet'],'checkpoint_persistence_verified':True,'exact_GPU_restore_performed':False,
            'publication_role':'completed_window_end','current_latest_recovery_required_before_phase_DEV_winner_advance':True}
        self.candidate_intent=deepcopy(intent)
        return self.capture

    def run_window(self,command):
        require(not self.inflight,'No overlapping native command')
        verify_merged_CPU_native_handoff(command,self.config,optimizer_updates=self.optimizer.updates,
            sampler_sha256=digest(self.schedule.state_dict()),guard=self.guard)
        plan=read_bound(command['coordinator_plan'],self.guard);owner=read_bound(command['feature_owner'],self.guard)
        manifest=read_bound(command['feature_manifest'],self.guard);before=deepcopy(self.schedule.state_dict())
        head=read_logical_durable_head(command['lifecycle_parent_head'])['head']
        require(head['identity_sha256']==digest(self.identity) and head['lifecycle']['last_sampler_sha256']==digest(before)
                and command['session']==owner['session']==self.config['session'] and manifest['owner']==command['feature_owner'],
                'Actual current durable head/source/session before first draw')
        completed=len(head['lifecycle']['completed_boundaries']);action=next_boundary_request(before,completed)
        validate_window_plan(before,action,plan)
        if self.config['owner_validation_only']:
            expected=self.context.validation_plan(self.schedule,completed_boundaries=completed,
                maximum_updates=self.config['maximum_validation_updates_per_window'])
        else:expected=self.context.plan(self.schedule,completed_boundaries=completed)
        require(digest(plan)==digest(expected),'Independent original full source plan')
        workspace=command['auxiliary_workspace'];marker=read_bound(workspace['workspace_marker'],self.guard)
        require(marker['kind']=='ASR18_fresh_future_owned_auxiliary_workspace_v1'
                and marker['publication_id']==command['checkpoint_intent']['id']
                and marker['plan']==command['coordinator_plan'] and marker['workspace']==workspace['workspace']
                and marker['created_before_any_producer'] is True,
                'Exact storage-reserved current auxiliary workspace before native output')
        root=Path(workspace['workspace']).resolve()
        require(root.is_relative_to(Path(self.config['checkpoint_storage_namespace']).resolve())
                and Path(workspace['workspace_marker']['path']).resolve()==root/'ownership.json',
                'No unowned native auxiliary output or old workspace adoption')
        self.phase_records=[];self.window_count+=1
        self.active_folder=root/'native';self.active_folder.mkdir(exist_ok=False)
        self.binding=MetadataCheckpointBinding(owner['augmentation_identity'],guard=self.guard)
        self.reader=PhaseAwareFeatureWindowReader(command['feature_owner'],command['feature_manifest'],guard=self.guard)
        self.coordinator=WindowCoordinator(self.schedule,self.optimizer,self.journal,plan,resolve_identity=self.context.resolve_identity)
        for planned in plan['updates']:
            index=self.coordinator.index
            def prepare(entry):return self.reader.update_record(index)
            def verify(entry,evidence):
                actual=read_bound(evidence,self.guard)
                require(all(actual[k]==entry[k] for k in ('native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','batch_id')),
                        'Exact closed CPU preparation transaction')
                return True
            def execute(entry,tx):
                rows=self.context.resolve_rows(tx.phase,tx.TR_draws,tx.replay_rows)
                require(digest(rows)==digest(entry['rows']),'Unchanged original source/full targets and exact replay')
                prepared=read_bound(self.reader.update_record(index),self.guard)
                budget=self.forecast.budget(rows=prepared['rows'],current_private_bytes=self._private('next_native'),update_in_process=self.native_in_process+1)
                path=self.active_folder/('native_'+str(self.optimizer.updates+1)+'_forecast.json');write_json(path,budget)
                self.inflight=True;native=None
                try:
                    def operation():
                        nonlocal native
                        native=native_update(self.model,self.teacher,self.optimizer,rows,entry['draws'],
                            feature_for_row=lambda i:self.reader.take_row(index,i,context=entry['contexts'][i],
                                native_transaction_sha256=tx.identity_digest,before_sampler_sha256=digest(tx.before)),
                            update_binding=self.reader.update_record(index),torch=self.torch,population_mean_U=self.schedule.means[tx.phase],
                            update=self.optimizer.updates+1,stage=self.stage,stop_check=self.guard,phase=tx.phase,feature_update_index=index,reference_original_head=False)
                        self._progress(native=native);return native
                    native,memory=self._run_phase('native_'+str(self.optimizer.updates+1),phase='native',increment=budget['incremental_bytes'],operation=operation)
                    self.forecast.observe(memory);self.native_in_process+=1
                    require(self.reader.prepared_payload(index)==native['augmentation'],'Actual consumed prepared payload before scientific commit')
                    return native
                except BaseException as error:
                    self.optimizer.poisoned=True;self._progress(native=native,error=repr(error));raise
                finally:self.inflight=False
            result=self.coordinator.consume(prepare_evidence=prepare,native_call=execute,verify_preparation=verify)
            self._progress(native=result['native'],committed=True)
        require(self.coordinator.index==len(plan['updates']),'Whole planned normal window before publication')
        capture=self._checkpoint(command)
        result={'kind':'ASR18_full_run_actual_resident_window_persistence_completion_v1','passed':True,'diagnostic_only':self.config['diagnostic_only'],
            'run_identity_sha256':digest(self.identity),'lifecycle_parent_head_sha256':command['lifecycle_parent_head']['sha256'],
            'window_index':command['window_index'],'actual_new_updates':len(plan['updates']),'global_new_updates':self.optimizer.updates,
            'coordinator_plan':command['coordinator_plan'],'feature_owner':command['feature_owner'],'feature_manifest':command['feature_manifest'],
            'latest_pointer':self.pointer,'checkpoints':[capture],'sampler_state_sha256':digest(self.schedule.state_dict()),
            'journal_prefix':self.journal.binding(),'native_forecast_high_water':self.forecast.state_dict(),
            'phase_records':deepcopy(self.phase_records),'all_checkpoint_persistence_verified':True,'exact_GPU_restores_deferred':True,
            'production_enabled':False,'qualified_checkpoint':False}
        path=self.active_folder/'completion.json';write_json(path,result);self.last_window=record(path)
        # No completed-window/native receipt arrays accumulate across the run.
        self.reader=None;self.coordinator=None;return self.last_window

    def coherent_idle(self,*,nonce,window_completion):
        require(not self.inflight and window_completion==self.last_window and type(nonce) is str and len(nonce)>=16,
                'Current complete window and exact parent nonce')
        self.guard();self.torch.cuda.synchronize();assert_model(self.model,self.optimizer,self.torch)
        require(all(p.grad is None for p in self.optimizer.parameters.values())
                and self.model._diagnostic_frozen_metrics['logical_current_matrix_bytes']==0,'No live native graph/gradient/streamed matrix')
        pointer=read_bound(self.pointer,self.guard);state=self.schedule.state_dict()
        require(pointer['committed_updates']==self.optimizer.updates and pointer['sampler_state_sha256']==digest(state)
                and pointer['journal_prefix']==self.journal.binding(),'Exact saved/live coherent prefix')
        from fastmss_owned_process_closure_v2 import identity
        return {'kind':'ASR18_resident_native_persistence_verified_idle_boundary_v2','owner':'/root/memory_diagnosis',
            'worker_source':record(self.config['worker_driver_source']['path']),'engine_source':record(__file__),'parent_source':self.config['parent_driver_source'],
            'producer':identity(),'request_nonce':nonce,'checkpoint_pointer':self.pointer,'checkpoint':pointer['checkpoint'],'payload':pointer['payload'],
            'phase':self.schedule.phase,'optimizer_updates':self.optimizer.updates,
            'committed_window_updates':read_bound(self.last_window,self.guard)['actual_new_updates'],
            'sampler_state_sha256':digest(state),'journal_prefix':self.journal.binding(),'window_completion':self.last_window,
            'all_window_updates_committed':True,'cuda_synchronized':True,'no_inflight_native_or_forward_hooks':True,
            'checkpoint_persistence_verified':True,'checkpoint_persistence_proof':self.last_persistence,
            'checkpoint_exact_restore_verified':False,'deferred_restore_required_before_overall_diagnostic_pass':True,
            'live_producer':True,'GPU_process_exit_claimed':False,'native_forecast_high_water':self.forecast.state_dict(),
            'production_enabled':False}

    def checkpoint_storage_release_ack(self,*,nonce,window_completion):
        """Same actual scoped writer/read/restore release invariant as engine6."""
        idle=self.coherent_idle(nonce=nonce,window_completion=window_completion)
        require(self.capture is not None and self.capture['pointer']==self.pointer and self.candidate_intent is not None
                and not self.checkpoint_writer_open and not self.checkpoint_reader_open and not self.checkpoint_restore_open,
                'Actual candidate writer/persistence/startup restore contexts returned')
        pointer=read_bound(self.pointer,self.guard)
        return {'kind':'ASR18_actual_native_checkpoint_mapping_release_ack_v1','producer':idle['producer'],
            'worker_source':idle['worker_source'],'engine_source':record(__file__),'request_nonce':nonce,
            'optimizer_updates':self.optimizer.updates,'checkpoint_pointer':self.pointer,'writer_intent':self.candidate_intent,
            'checkpoint':pointer['checkpoint'],'payload':pointer['payload'],'checkpoint_persistence_proof':self.last_persistence,
            'native_work_quiescent':True,'checkpoint_mappings_released':True,'candidate_payload_handles_released':True,
            'checkpoint_write_inflight':False,'restore_in_progress':False,'writer_context_returned':True,
            'persistence_reader_context_returned':True,'startup_restore_context_returned':True,
            'release_scope':'candidate training-checkpoint files, excludes immutable base-model bank',
            'writer_implementation':record(Path(__file__).with_name('bpe_streaming_training_state_v1.py')),
            'streamed_file_implementation':record(Path(__file__).with_name('bpe_streaming_checkpoint_v1.py')),
            'current_sampler_sha256':idle['sampler_state_sha256'],'current_journal_prefix':idle['journal_prefix'],
            'production_enabled':False,'qualified':False}

    def close(self):self.context.close()
