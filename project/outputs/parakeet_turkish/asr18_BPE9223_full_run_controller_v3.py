"""Inactive metadata-lifetime successor; scientific callbacks inherit frozen2.

Only preparation-frame lifetime and run-loop local lifetime change here. The
old exact source-bound storage/lifecycle/proof methods retain their original
code objects and inspect.getsourcefile provenance. No abnormal resume authority
is provided by this module; it requires its separate reviewed transition.
"""
import gc,time
from pathlib import Path
from copy import deepcopy
import asr18_BPE9223_full_run_controller_v2 as frozen
from asr18_BPE9223_full_run_controller_v2 import (
    record,require,write_json,read_json,read_bound,digest,bridge_plan,verify_resident_idle,
    FullRunLifecycle,PackedDurableLifecycleAdapter)
import secrets,shutil

class FullOwner(frozen.FullOwner):
    def _prepare_metadata(self,plan,index):
        self.check()
        verified=self._verify_prior_prefix_before_reservation() if index>0 else None
        intent=self.store.reserve_publication(publication_role='completed_window_end');self.current_intent=intent
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
        from asr18_BPE9223_CPU_window_configuration_v2 import derive_contract,derive_configuration
        cpu=root/'CPU';template=self.config['CPU_template_configuration']
        sc=self._new(root/'CPU_source_contract.json',derive_contract(template,record(bp),self.parent.check))
        proposal=self._new(root/'CPU_window_configuration.json',derive_configuration(template,sc,index,intent['id'],str(cpu/'payload'),self.parent.check))
        arguments=['--config',proposal['path'],'--ROOT-review',self.config['ROOT_CPU_execution_review']['path'],'--folder',str(cpu/'payload')]
        if index>0:
            boundary=read_bound(self.prior_boundary,self.parent.check)
            self.verify_prior_boundary(boundary)
            require(boundary==verified['boundary'] and self.durable.head_pin==boundary['durable_head']
                and digest(self.schedule.state_dict())==verified['sampler_sha256'],
                'After new bridge, recheck SAME previously verified old head/capture/sampler')
            pointer=read_bound(self.current_capture['pointer'],self.parent.check)
            require(pointer==verified['pointer'],'No prior pointer mutation between archive verification and CPU handoff')
            if self.gpu is not None:
                self.gpu.check()
                verify_resident_idle(read_bound(self.current_idle,self.parent.check),pointer=pointer,
                    proof_binding=self.current_idle,phase=verified['phase'],
                    guard=self.parent.check)
            else:
                require(self.current_ack is not None,'Closed native handoff requires actual current recovery ACK')
                self._verify_closed(self.closed_native);self._verify_closed(self.closed_recovery)
            attestation={'kind':'FastMSS_native_owner_prior_window_CPU_handoff_v1','owner':OWNER,'passed':True,
                'boundary_sha256':digest(boundary),'actual_all_state_prefix_checkpoint_verified':True,
                'bridge':record(bp),'session':self.config['session'],'parent_source':self.parent_source,
                'durable_head':self.durable.head_pin,'current_durable_owner_head_and_persistence_prefix_verified':True,
                'current_checkpoint_capture':self.current_capture,'source':record(__file__),
                'saved_sampler_sha256':verified['sampler_sha256'],'historical_fresh_GPU_restore_claimed':False}
            ap=root/'prior_boundary_verification.json';self._new(ap,attestation)
            ab=root/'prior_boundary_verification_binding.json';self._new(ab,record(ap))
            # Exact bridge already binds this freshly reverified prior boundary;
            # generic CPU3 reconstructs it independently through the context callback.
            # Retain the actual handoff attestation as archive metadata, not a fake CLI proof.
        # No parsed bridge, prior verification dictionaries or decoded pointer
        # escapes this helper frame. Only original immutable pins/scalars do.
        return {'phase':plan['phase'],'plan_pin':plan_pin,'window_index':index,
            'intent':intent,'workspace':workspace,'bridge_pin':record(bp),
            'CPU_contract':sc,'CPU_folder':str(cpu),'arguments':arguments}

    def _execute_prepared_CPU(self,prepared):
        # Full collection changes metadata lifetime only. The SAME actual64M
        # entry and live CPU gates remain authoritative; no release is assumed.
        before=self.parent.check()['worker_private_memory']['private_bytes']
        collected=gc.collect()
        after=self.parent.check()['worker_private_memory']['private_bytes']
        root=Path(prepared['workspace']['workspace']);cpu=Path(prepared['CPU_folder'])
        self._new(root/'metadata_builder_frame_release.json',{
            'kind':'ASR18_actual_metadata_builder_frame_release_v1','source':record(__file__),
            'parsed_bridge_verified_boundary_pointer_frames_returned':True,
            'prior_run_plan_completion_response_packet_locals_released':True,
            'full_collection_generation':2,'objects_collected':collected,
            'actual_before_private_bytes':before,'actual_after_private_bytes':after,
            'CPU_current_cap_bytes':64000000,'release_or_resource_credit_assumed':False})
        with self.parent.CPU_phase():
            self._admit_CPU_family(root/'CPU_remaining_family_admission.json')
            proof=self.run_CPU(self.config,self.config['CPU_controller_source']['path'],prepared['arguments'],
                cpu,guard=self.parent.check if self.gpu is None else self.gpu.check,report_relative='payload/report.json')
            report=read_json(cpu/'payload/report.json');self._verify_aggregate_CPU_report(report,prepared['CPU_contract'])
        packet={'window_index':prepared['window_index'],'phase':prepared['phase'],
            'coordinator_plan':prepared['plan_pin'],'feature_owner':report['owner'],'feature_manifest':report['manifest'],
            'session':self.config['session'],'lifecycle_parent_head':self.durable.head_pin,
            'checkpoint_intent':prepared['intent'],'auxiliary_workspace':prepared['workspace']}
        self.current_packet=packet;self.current_CPU0=proof;self.current_CPU_report=record(cpu/'payload/report.json')
        self.current_bridge=prepared['bridge_pin'];self.current_workspace=prepared['workspace']
        self.current_original_plan=prepared['plan_pin']
        return packet

    def run(self):
        while True:
            self.check()
            owner_reserve=(self.config['maximum_CPU_phase_seconds']+self.config['maximum_native_window_seconds']
                +self.config['maximum_archive_phase_seconds']+self.config['maximum_checkpoint_phase_seconds']
                +self.config['maximum_cleanup_phase_seconds']+self.config['maximum_native_exit_seconds']
                +self.config['maximum_recovery_phase_seconds'])
            require(0<owner_reserve<self.config['maximum_owner_wall_seconds'],'One complete future window plus coherent owner closure reserve')
            if (self.administrative_maximum_new_updates is not None and self.current_capture is not None
                and self.schedule.committed_updates()-self.invocation_initial_updates>=self.administrative_maximum_new_updates
                and self.schedule.committed_updates()<self.config['planned_native_updates']):
                return self._normal_owner_pause()
            if (self.current_capture is not None and self.schedule.committed_updates()<self.config['planned_native_updates']
                and time.monotonic()-self.began+owner_reserve>=self.config['maximum_owner_wall_seconds']):
                return self._normal_owner_pause()
            require(time.monotonic()-self.began<self.config['maximum_owner_wall_seconds'],'Reviewed outer owner wall budget')
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
                from asr18_BPE9223_lifecycle_v3 import official_DEV
                score=official_DEV(read_bound(report,self.parent.check),self.current_capture['pointer'],read_bound(self.identity['scorer_policy'],self.parent.check))
                old=self.lifecycle.state_dict()['winner'];change=score['strict_all_six_passed'] and (old is None or score['word_errors']['tr']<old['word_errors']['tr'])
                winner=self.preserve_winner({'replace':change,'old_winner':old,'candidate_pointer':self.current_capture['pointer'],'DEV':report,'score':score})
                self.durable.acknowledge_DEV_and_winner(report=report,winner=winner);self._refresh_boundary();continue
            require(action['kind']=='bounded_training_window','Unsupported tail preserved, no guessed recovery')
            self.before_window=deepcopy(self.schedule.state_dict());count=len(self.lifecycle.state_dict()['completed_boundaries'])
            plan=(self.context.validation_plan(self.schedule,completed_boundaries=count,maximum_updates=4)
                if self.config['owner_validation_only'] else self.context.plan(self.schedule,completed_boundaries=count))
            if (self.administrative_maximum_new_updates is not None and self.current_capture is not None
                and self.schedule.committed_updates()-self.invocation_initial_updates+len(plan['updates'])>self.administrative_maximum_new_updates):
                return self._normal_owner_pause()
            prepared=self._prepare_metadata(plan,self.last_window_index+1)
            # The exact plan is already bound in registered AUX. Drop its parsed
            # caller copy BEFORE entering the CPU frame, then reload that exact
            # pin only when archive/publish needs its values.
            del plan
            packet=self._execute_prepared_CPU(prepared)
            if self.gpu is None:self._start_native(packet)
            response=self.gpu.command('run_prepared_window',window=packet);self.current_response=response
            self.current_completion=response['window_completion'];completion=read_bound(self.current_completion,self.parent.check)
            self.current_capture=completion['checkpoints'][0];self.current_release=response['checkpoint_storage_release_ack']
            self.current_idle=self.gpu.seal_idle(response)
            self._cleanup(completion)
            plan=read_bound(prepared['plan_pin'],self.parent.check)
            self._archive_and_publish(plan,self.current_completion);self.last_window_index+=1
            del plan,prepared,completion,response,packet
            gc.collect()
            if self.config['owner_validation_only']:
                actual=self.schedule.committed_updates()
                require(actual in (4,8),'Same original productionbody, two actual complete4-update windows')
                if actual==4 and self.actual_owner_invocation_count==1:return self._normal_owner_pause()
                if actual==8:
                    self._recover();break
            self._refresh_boundary()

        if self.gpu is not None:self._recover()
        # Actually reopen the packed authority after current exact recovery.
        restored=self.lifecycle.state_dict();self.durable.close()
        lifecycle=FullRunLifecycle(self.schedule,identity=self.identity,callbacks=self.lifecycle.callbacks,
            qualified_winner=self.config['qualified_source_winner'],restored=restored,restore_proof=self.current_ack)
        self.lifecycle=lifecycle
        self.durable=self._reopen_completed_durable(lifecycle)
        require(self.durable.state_dict()['pending']['publications']==0,'Actual current recovery and durable reopen0')
        result={'kind':'ASR18_actual_full_run_owner_pipeline_validation_v1','passed':True,'producer':self.producer,
            'worker_source':record(__file__),'parent_source':self.parent_source,'run_identity_sha256':digest(self.identity),
            'actual_owner_invocation_count':self.actual_owner_invocation_count,'actual_normal_resume_request':self.resume_request,
            'actual_optimizer_updates':self.schedule.committed_updates(),'completed_passes':len(self.lifecycle.state_dict()['completed_boundaries']),
            'current_checkpoint':self.current_capture,'current_native_process0':self.closed_native,'current_recovery_process0':self.closed_recovery,
            'current_recovery_report':self.current_recovery_report,'current_persistence_prefix_coverage':self.current_coverage,
            'durable_current_recovery_ack':self.current_ack,'durable_reopened_head':self.durable.head_pin,
            'actual_obsolete_worker_control_archives':self.worker_control_archive_result,
            'worker_control_pool':self.parent.control_pool.snapshot_binding(),
            'qualified_source_winner_preserved':self.lifecycle.state_dict()['winner']==self.config['qualified_source_winner'],
            'owner_validation_only':self.config['owner_validation_only'],'production_source_eligible':not self.config['owner_validation_only'],
            'no_fake_pass_or_DEV_barrier':True,'production_enabled':False,'wall_seconds':time.monotonic()-self.began}
        write_json(self.folder/'report.json',result);return result

    def _reopen_completed_durable(self,lifecycle):
        return PackedDurableLifecycleAdapter(lifecycle,self.root/'durable',
            policy=self.config['durable_policy'],proofs=self.proofs,create=False,resume_proof=self.current_ack)
