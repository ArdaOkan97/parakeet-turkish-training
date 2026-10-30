"""Opt-in finite full-DATA native worker engine; no automatic production.

The engine can retain a synchronized coherent model across CPU preparation.
Its caller owns the fresh system guard, actual children, file commands and
combined resident admission. No decoder/frontend/augmenter runs in this module.
"""
from copy import deepcopy
import json, os, time
from pathlib import Path
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import CommitJournal, digest
from bpe_training_schedule_v1 import WEAK
from bpe_production_window_coordinator_v4 import WindowCoordinator
from bpe_final_DATA_schedule_context_v1 import ScheduleContext, normalize_schedule_state
from bpe_streaming_training_state_v1 import checkpoint as save_state, restore as restore_state
from bpe_broad_production_v3 import assert_model, prohibit_model_transforms
from asr18_precomputed_native_v2 import native_update
from asr18_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
from asr18_calibrated_geometry_forecast_v6 import CalibratedGeometryForecast
from asr18_separated_worker_forecast_v1 import phase_projection
from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
from asr18_live_state_fingerprint_v1 import live_state_fingerprint, require_exact_live_restore
from resource_watchdog_v2 import sample_memory


class NativeWindowEngine:
    def __init__(self, model, processor, teacher, optimizer, provenance, bootstrap_snapshot,
                 folder, config, *, torch, stage, admit, guard):
        require(config['production_enabled'] is False and config['diagnostic_only'] is True
                and type(config['maximum_windows']) is int and 1 <= config['maximum_windows'] <= 2
                and type(config['maximum_total_native_updates']) is int and 1 <= config['maximum_total_native_updates'] <= 40,
                'Only an explicitly bounded full-DATA diagnostic; no production launch')
        from asr18_temporal_tiled_subsampling_v2 import apply as validated_conv_only
        require(record(Path(__file__).with_name('asr18_temporal_tiled_subsampling_v2.py')) == config['validated_conv_only_source'],'Exact validated frozen CNN component source')
        provenance['validated_conv_only_CNN_lifetime_policy'] = validated_conv_only(model,torch=torch,guard=guard)
        self.model = model; self.teacher = teacher; self.optimizer = optimizer; self.torch = torch
        self.folder = Path(folder); self.config = config; self.provenance = provenance
        self.stage = stage; self.admit = admit; self.guard = guard
        self.context = ScheduleContext(config['planning_arguments'], guard=guard)
        require(digest(bootstrap_snapshot) == digest(self.context.snapshot) and optimizer.updates == 0,
                'Bootstrap must retain original BPE64250/replay RNG, never diagnostic20/21')
        self.schedule = self.context.new_schedule(); self.forecast = CalibratedGeometryForecast(**config['forecast_calibration'], guard=guard)
        self.journal = CommitJournal(self.folder/'original_authoritative_commits.jsonl')
        self.binding = None; self.inflight = False; self.window_count = 0
        self.process_initial_updates = 0; self.native_in_process = 0; self.history = []; self.pointer = None
        self.coordinator = None; self.reader = None; self.last_pre_save_fingerprint = None; self.phase_records = []; self.last_persistence = None
        prohibit_model_transforms(model, torch); model._broad_processor = processor

    def _private(self, label):
        value = sample_memory(os.getpid(), label)
        require(value['paired_sampling_within_one_second'] and value['system_memory']['available']
                and value['worker_private_memory']['available'], 'Actual fresh paired phase telemetry required')
        return value['worker_private_memory']['private_bytes']

    def _run_phase(self, name, *, phase, increment, operation):
        self.guard(); before = self._private(name+'_before')
        projection = phase_projection(self.config['entire_worker_forecast'], phase=phase,
                                     current_private_bytes=before, increment_bytes=increment)
        self.admit(name, increment); monitor = PhasePrivateMonitor(); monitor.start(); began = time.monotonic()
        returned = False
        try: result = operation(); returned = True
        finally:
            proof = monitor.finish(); proof['wall_seconds'] = time.monotonic()-began
            covered = proof['private_peak_bytes'] <= before+increment
            write_json(self.folder/(name+'_memory.json'), {'before_private_bytes': before, 'increment_bytes': increment,
                'projection': projection, 'actual': proof, 'covered': covered,
                'margin_bytes': before+increment-proof['private_peak_bytes']})
            self.phase_records.append({'name':name,'phase':phase,'memory_record':record(self.folder/(name+'_memory.json')),
                'wall_seconds':proof['wall_seconds'],'operation_returned':returned,'forecast_covered':covered})
        require(covered, 'Prospective '+phase+' forecast missed; preserve partial proof and stop without retry')
        return result, proof

    def _progress(self, *, native=None, committed=False, error=None):
        value = {'kind': 'ASR18_full_DATA_bounded_actual_native_progress_v1',
            'actual_optimizer_updates': self.optimizer.updates, 'native_updates_this_process': self.optimizer.updates-self.process_initial_updates,
            'native_return_complete': native is not None, 'journal_sampler_committed': committed,
            'optimizer_poisoned': self.optimizer.poisoned, 'schedule_poisoned': self.schedule.poisoned,
            'clocks': [base+self.optimizer.updates for base in (64250, 2000, 62250)],
            'error': error, 'production_enabled': False}
        write_json(self.folder/'native_progress.json', value)

    def restore_parent(self, pointer_binding, *, augmentation_identity):
        from fastmss_phase_aware_feature_window_v4 import read_bound
        pointer = read_bound(pointer_binding, self.guard)
        require(pointer['kind'] == 'ASR18_production_window_completed_checkpoint_pointer_v2'
                and pointer['all_window_updates_committed'] is True and pointer['diagnostic_only'] is True
                and 1 <= pointer['committed_updates'] <= 20, 'Actual first-window coherent source required')
        self.binding = MetadataCheckpointBinding(augmentation_identity, guard=self.guard)
        def restore():
            from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
            from asr18_checkpoint_next_draw_v1 import next_draw_packet
            restored = restore_state(pointer['checkpoint'], torch=self.torch, model=self.model, optimizer=self.optimizer,
                identity=pointer['identity'], augmentation=self.binding,
                new_journal_path=self.folder/'restored_parent_authoritative_prefix.jsonl', diagnostic_only=True, guard=self.guard)
            state, restored_journal, branch = restored
            self.schedule.load_state_dict(normalize_schedule_state(state))
            caller = pointer['identity']['caller_identity']
            expected = normalize_fingerprint(read_bound(caller['pre_save_fingerprint'], self.guard))
            require_exact_live_restore(expected, live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard))
            actual_next = next_draw_packet(self.schedule,self.context,optimizer_updates=self.optimizer.updates,guard=self.guard)
            require(digest(actual_next)==digest(read_bound(caller['next_draw_packet'],self.guard)),
                    'Startup fallback restore must match actual PRE-save next draw, contexts and full source rows')
            return restored
        (state, journal, branch), memory = self._run_phase('restore_actual_first_window', phase='startup_restore', increment=350_000_000, operation=restore)
        self.journal = journal
        require(self.optimizer.updates == self.schedule.committed_updates() == pointer['committed_updates']
                and digest(state) == pointer['sampler_state_sha256'], 'Actual first-window all-state/frontier restore required')
        caller = pointer['identity']['caller_identity']
        require(caller['session'] == self.config['session'], 'Fallback cannot change source-bound diagnostic session')
        self.forecast.load_state_dict(caller['native_forecast_high_water'])
        self.process_initial_updates = self.optimizer.updates; self.window_count = 1
        write_json(self.folder/'actual_parent_restore.json', {'passed': True, 'pointer': pointer_binding,
            'branch': branch, 'memory': memory, 'actual_state_sha256': digest(state),
            'all207_threeAdam_all4RNG_match_actual_pre_save':True,'next_draw_packet_match':True,
            'restore_occurred_only_at_startup_before_native':True,'production_enabled': False})

    def _checkpoint(self, window_folder, window_binding, owner_binding):
        from asr18_streamed_persistence_fingerprint_v1 import verify_persistence
        from asr18_checkpoint_next_draw_v1 import next_draw_packet
        before = self.schedule.state_dict(); last = self.coordinator.plan['updates'][self.coordinator.index-1]
        update = self.optimizer.updates
        caller_identity = {'kind':'ASR18_phase_separated_bounded_window_caller_identity_v2',
            'session':self.config['session'],'config':record(self.config['_path']),'engine':record(__file__),
            'native':record(Path(__file__).with_name('asr18_precomputed_native_v2.py')),
            'feature_window':window_binding,'feature_owner':owner_binding,
            'native_forecast_high_water':self.forecast.state_dict(),'production_enabled':False,
            'exact_GPU_restore_deferred_until_actual_native_exit':True}
        def publish_and_verify():
            self.last_pre_save_fingerprint = live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard)
            fingerprint_path = window_folder/('checkpoint_'+str(update)+'_actual_pre_save_fingerprint.json')
            next_path = window_folder/('checkpoint_'+str(update)+'_actual_next_draw_packet.json')
            require(not fingerprint_path.exists() and not next_path.exists(),'Fresh independent PRE-save evidence')
            write_json(fingerprint_path,self.last_pre_save_fingerprint)
            write_json(next_path,next_draw_packet(self.schedule,self.context,optimizer_updates=update,guard=self.guard))
            caller_identity.update(pre_save_fingerprint=record(fingerprint_path),next_draw_packet=record(next_path))
            def write_checkpoint(state,identity):
                require(digest(state)==digest(before),'Exact pre-writer actual sampler')
                return save_state(window_folder/('after_update_'+str(update)+'.state.json'),torch=self.torch,
                    optimizer=self.optimizer,state=state,provenance=self.provenance,identity=identity,
                    augmentation=self.binding,journal=self.journal,diagnostic_only=True,guard=self.guard)
            pointer = self.coordinator.publish_checkpoint(write_coherent_checkpoint=write_checkpoint,
                pointer_path=window_folder/'latest_coherent_checkpoint.json',checkpoint_identity=caller_identity,diagnostic_only=True)
            immutable = window_folder/('checkpoint_'+str(update)+'_immutable_pointer.json')
            require(not immutable.exists(),'Immutable pointer retained for every deferred restore')
            write_json(immutable,pointer); self.pointer = record(immutable)
            proof = verify_persistence(pointer['checkpoint'],expected_pre_save=self.last_pre_save_fingerprint,
                identity=pointer['identity'],sampler_state=before,journal_prefix=pointer['journal_prefix'],
                diagnostic_only=True,guard=self.guard)
            after = live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard)
            require_exact_live_restore(self.last_pre_save_fingerprint,after)
            require(digest(self.schedule.state_dict())==digest(before) and self.journal.binding()==pointer['journal_prefix'],
                    'Read-only persistence check must leave actual frontiers/prefix untouched')
            proof.update(checkpoint_pointer=self.pointer,pre_save_fingerprint=record(fingerprint_path),
                next_draw_packet=record(next_path),live_after_state_sha256=after['state_sha256'],
                live_unchanged_after_persistence=True,worker_source=record(self.config['worker_driver_source']),
                engine_source=record(__file__),deferred_restore_required_before_overall_diagnostic_pass=True,
                actual_GPU_restore_performed=False)
            proof_path = window_folder/('checkpoint_'+str(update)+'_persistence_verified.json')
            write_json(proof_path,proof);self.last_persistence=record(proof_path)
            return pointer,proof
        (pointer,proof),memory = self._run_phase('window'+str(self.window_count)+'_checkpoint'+str(update),
            phase='checkpoint_write',increment=64_000_000,operation=publish_and_verify)
        self._resume_reader(pointer,window_binding,owner_binding,window_folder,last)
        return {'pointer':self.pointer,'checkpoint':pointer['checkpoint'],'write_and_persistence_memory':memory,
            'checkpoint_persistence_proof':self.last_persistence,'pre_save_fingerprint':proof['pre_save_fingerprint'],
            'next_draw_packet':proof['next_draw_packet'],'checkpoint_persistence_verified':True,
            'exact_GPU_restore_performed':False,'deferred_restore_required_before_overall_diagnostic_pass':True}

    def _resume_reader(self, pointer, manifest, owner, window_folder, last):
        from fastmss_phase_aware_feature_window_v4 import PhaseAwareFeatureWindowReader, read_bound
        bound_owner = read_bound(owner, self.guard); count = self.coordinator.index
        value = {'kind': 'FastMSS_phase_aware_feature_prefix_resume_v1', 'bridge': bound_owner['bridge'],
            'phase': self.schedule.phase, 'committed_window_updates': count, 'optimizer_updates': self.optimizer.updates,
            'last_transaction_sha256': last['native_transaction_sha256'], 'sampler_state_sha256': digest(self.schedule.state_dict()),
            'journal_prefix': pointer['journal_prefix'], 'coherent_checkpoint_pointer': self.pointer, 'checkpoint': pointer['checkpoint']}
        resume_path = window_folder/('reader_resume_'+str(self.optimizer.updates)+'.json'); write_json(resume_path, value)
        def verify(resume):
            require(resume == value and digest(resume) == digest(value)
                    and digest(self.schedule.state_dict()) == value['sampler_state_sha256']
                    and self.optimizer.updates == value['optimizer_updates'], 'Actual unchanged live sampler and persistence-verified saved prefix required')
            return True
        self.reader = PhaseAwareFeatureWindowReader(owner, manifest, guard=self.guard,
                    resume_binding=record(resume_path), verify_checkpoint=verify)

    def run_window(self, command):
        from fastmss_phase_aware_feature_window_v4 import PhaseAwareFeatureWindowReader, read_bound
        require(not self.inflight and self.window_count < self.config['maximum_windows'], 'No concurrent or extra native window')
        self.guard(); plan = read_bound(command['coordinator_plan'], self.guard)
        owner = read_bound(command['feature_owner'], self.guard); manifest = read_bound(command['feature_manifest'], self.guard)
        require(plan['phase'] == self.schedule.phase == WEAK and plan['initial_committed_updates'] == self.optimizer.updates
                and plan['initial_sampler_sha256'] == digest(self.schedule.state_dict())
                and self.optimizer.updates+len(plan['updates']) <= self.config['maximum_total_native_updates'],
                'Window must start at actual source sampler; phase/DEV barriers cannot be skipped')
        require(command['session'] == self.config['session'] and owner['session'] == self.config['session']
                and manifest['owner'] == command['feature_owner'], 'Exact source-bound finite session and actual CPU owner')
        self.binding = MetadataCheckpointBinding(owner['augmentation_identity'], guard=self.guard)
        self.reader = PhaseAwareFeatureWindowReader(command['feature_owner'], command['feature_manifest'], guard=self.guard)
        self.coordinator = WindowCoordinator(self.schedule, self.optimizer, self.journal, plan, resolve_identity=self.context.resolve_identity)
        self.window_count += 1; window_folder = self.folder/('window_'+str(self.window_count)); window_folder.mkdir(exist_ok=False)
        began = time.monotonic(); checkpoints = []; phase_start = len(self.phase_records); audio_seconds = 0.
        for planned in plan['updates']:
            self.guard(); cursor = self.schedule.active_cursor
            require(cursor.draws == 0 or cursor.draws % cursor.population != 0,
                    'Full-pass DEV barrier reached; preserve checkpoint and stop before next pass draw')
            index = self.coordinator.index
            def prepare(entry): return self.reader.update_record(index)
            def verify_preparation(entry, evidence):
                prepared = read_bound(evidence, self.guard)
                require(all(prepared[key] == entry[key] for key in ('native_transaction_sha256', 'before_sampler_sha256', 'after_sampler_sha256', 'batch_id'))
                        and digest(prepared['rows']) == digest(read_bound(self.reader.update_record(index), self.guard)['rows']),
                        'Actual closed prepared update must match exact planned transaction')
                return True
            def native_call(entry, tx):
                # Resolve the original retained DB/cache again, independently
                # of CPU prepared rows and the private coordinator plan.
                rows = self.context.resolve_rows(tx.phase, tx.TR_draws, tx.replay_rows)
                require(digest(rows) == digest(entry['rows']), 'Full original raw rows/labels/cache differ from preparation')
                prepared = read_bound(self.reader.update_record(index), self.guard)
                current = self._private('prospective_full_DATA_native_before')
                budget = self.forecast.budget(rows=prepared['rows'], current_private_bytes=current,
                                               update_in_process=self.native_in_process+1)
                write_json(window_folder/('native_'+str(self.optimizer.updates+1)+'_forecast.json'), budget)
                native = None; self.inflight = True
                try:
                    def operation():
                        nonlocal native
                        native = native_update(self.model, self.teacher, self.optimizer, rows, entry['draws'],
                            feature_for_row=lambda i: self.reader.take_row(index, i, context=entry['contexts'][i],
                                native_transaction_sha256=tx.identity_digest, before_sampler_sha256=digest(tx.before)),
                            update_binding=self.reader.update_record(index), torch=self.torch,
                            population_mean_U=self.schedule.means[tx.phase], update=self.optimizer.updates+1,
                            stage=self.stage, stop_check=self.guard, phase=tx.phase, feature_update_index=index)
                        self._progress(native=native)
                        return native
                    native, memory = self._run_phase('full_DATA_native_update_'+str(self.optimizer.updates+1),
                        phase='native', increment=budget['incremental_bytes'], operation=operation)
                    self.forecast.observe(memory); self.native_in_process += 1
                    require(self.reader.prepared_payload(index) == native['augmentation'], 'Consumed full payload differs before sampler commit')
                    return native
                except BaseException as error:
                    self.optimizer.poisoned = True; self._progress(native=native, error=repr(error)); raise
                finally: self.inflight = False
            result = self.coordinator.consume(prepare_evidence=prepare, native_call=native_call, verify_preparation=verify_preparation)
            self._progress(native=result['native'], committed=True); self.history.append(result['native'])
            audio_seconds += sum(row['samples']/16000 for row in read_bound(self.reader.update_record(index),self.guard)['rows'])
            if self.coordinator.index in (1, 10, len(plan['updates'])):
                checkpoints.append(self._checkpoint(window_folder, command['feature_manifest'], command['feature_owner']))
        require(self.coordinator.index == len(plan['updates']) and self.pointer is not None, 'Complete bounded window with actual final coherent checkpoint required')
        phase_records = self.phase_records[phase_start:]; elapsed = time.monotonic()-began
        phase_seconds = {phase:sum(item['wall_seconds'] for item in phase_records if item['phase']==phase)
                         for phase in ('native','checkpoint_write','startup_restore')}
        result = {'kind': 'ASR18_full_DATA_bounded_native_window_completion_v1', 'passed': True,
            'window_index': self.window_count-1, 'actual_new_updates': len(plan['updates']), 'global_new_updates': self.optimizer.updates,
            'coordinator_plan': command['coordinator_plan'], 'feature_owner': command['feature_owner'], 'feature_manifest': command['feature_manifest'],
            'latest_pointer': self.pointer, 'checkpoints': checkpoints, 'sampler_state_sha256': digest(self.schedule.state_dict()),
            'journal_prefix': self.journal.binding(), 'native_forecast_high_water': self.forecast.state_dict(),
            'wall_seconds': elapsed, 'phase_records':phase_records,'phase_seconds':phase_seconds,
            'worker_metadata_and_admission_overhead_seconds':max(0.,elapsed-sum(phase_seconds.values())),
            'actual_student_input_audio_seconds_including_replay':audio_seconds,
            'actual_native_RTF':phase_seconds['native']/audio_seconds,
            'actual_window_audio_hours_per_wall_hour':audio_seconds/elapsed,
            'production_enabled': False, 'qualified_checkpoint': False,
            'all_checkpoint_persistence_verified':True,'exact_GPU_restores_deferred':True,
            'overall_diagnostic_pass_before_separate_restore':False}
        write_json(window_folder/'completion.json', result); return record(window_folder/'completion.json')

    def coherent_idle(self, *, nonce, window_completion):
        require(type(nonce) is str and len(nonce) >= 16 and not self.inflight and self.coordinator is not None
                and self.coordinator.index == len(self.coordinator.plan['updates']) and not self.optimizer.poisoned,
                'Only final committed whole-window coherent state can become resident idle')
        self.guard(); self.torch.cuda.synchronize(); assert_model(self.model, self.optimizer, self.torch)
        require(self.model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0
                and all(parameter.grad is None for parameter in self.optimizer.parameters.values()),
                'No in-flight frozen forward materialization or live gradients at idle')
        state = self.schedule.state_dict(); pointer = json.loads(Path(self.pointer['path']).read_text(encoding='utf-8'))
        require(pointer['all_window_updates_committed'] is True and pointer['committed_updates'] == self.optimizer.updates
                and pointer['sampler_state_sha256'] == digest(state) and pointer['journal_prefix'] == self.journal.binding(),
                'Current coherent state/prefix must match exact returned checkpoint before idle')
        import psutil
        process = psutil.Process()
        return {'kind': 'ASR18_resident_native_persistence_verified_idle_boundary_v2', 'owner': '/root/memory_diagnosis',
            'worker_source': record(self.config['worker_driver_source']), 'engine_source': record(__file__),
            'parent_source': self.config['parent_driver_source'],
            'producer': {'pid': os.getpid(), 'ppid': os.getppid(), 'create_time': process.create_time()},
            'request_nonce': nonce, 'checkpoint_pointer': self.pointer, 'checkpoint': pointer['checkpoint'], 'payload': pointer['payload'],
            'phase': self.schedule.phase, 'optimizer_updates': self.optimizer.updates, 'committed_window_updates': self.coordinator.index,
            'sampler_state_sha256': digest(state), 'journal_prefix': self.journal.binding(), 'window_completion': window_completion,
            'all_window_updates_committed': True, 'cuda_synchronized': True, 'no_inflight_native_or_forward_hooks': True,
            'checkpoint_persistence_verified':True,'checkpoint_persistence_proof':self.last_persistence,
            'checkpoint_exact_restore_verified':False,'deferred_restore_required_before_overall_diagnostic_pass':True,
            'live_producer':True,'GPU_process_exit_claimed':False,
            'native_forecast_high_water': self.forecast.state_dict(), 'production_enabled': False}

    def close(self):
        self.context.close()
