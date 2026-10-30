"""Unarmed phase-aware production window/state coordination.

No model, decoder, download, frontend or native import occurs here. The caller
must separately admit actual producers/native work and validate the finalized
population. This module never treats a preparation as a scientific commit.
"""
from copy import deepcopy
import json, threading
from pathlib import Path
from bpe_broad_production_state_v1 import canonical, digest, require
from bpe_training_schedule_v1 import WEAK, CLEANER, COMPLETE
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json

CAPACITY=512*1024**2
METADATA_RESERVE=8*1024**2
MAX_UPDATES=20
ROOT=Path(r'E:\gpu_training_scratchpad')

def serial():
    require(threading.current_thread() is threading.main_thread(), 'Serial window/cursor use only')

def binding(value):
    require(type(value) is dict and set(value)=={'path','bytes','sha256'} and type(value['bytes']) is int and value['bytes']>0
            and record(value['path'])==value, 'Exact nonempty source-bound evidence required')

def make_plan(schedule, *, clone_schedule, resolve_identity, resolve_rows, feature_bytes_upper_bound,
              finalized_population, verify_finalized_population, maximum_updates=MAX_UPDATES):
    """Plan private schedule copies, preserving every authoritative frontier.

    feature_bytes_upper_bound covers all features/masks/files for an update,
    using declared conservative geometry. A too-large first update stops before
    a consumed draw; subsequent capacity stops make a shorter complete window.
    The actual sink still enforces the aggregate hard capacity independently.
    """
    serial();binding(finalized_population)
    require(verify_finalized_population(finalized_population) is True, 'Final population may not be bypassed by a partial mean')
    require(type(maximum_updates) is int and 1<=maximum_updates<=MAX_UPDATES and not schedule.poisoned
            and schedule.phase!=COMPLETE,'Finite nonempty unpoisoned phase window required')
    before=deepcopy(schedule.state_dict());private=clone_schedule()
    require(private is not schedule and private.replay is not schedule.replay,'Independent planning cursor/replay objects required')
    private.load_state_dict(deepcopy(before));phase=private.phase;updates=[];payload_budget=0
    try:
        for index in range(maximum_updates):
            if private.phase!=phase:break # A phase change always starts a new clean/weak producer.
            tx=private.propose(optimizer_updates=private.committed_updates(),resolve_TR=resolve_identity)
            try:
                rows=resolve_rows(phase,tuple(deepcopy(tx.TR_draws)),tuple(deepcopy(tx.replay_rows)))
                require(type(rows) is list and len(rows)==len(tx.TR_draws)+2 and 3<=len(rows)<=16,'Full tail plus exactly2 replay rows required')
                identities=tx.TR_identity+list(tx.replay_rows)
                require([(r['source_id'],r['language']) for r in rows]==[(r['source_id'],r['language']) for r in identities],
                        'Resolved full rows differ from independent phase/ordinal/replay identities')
                size=feature_bytes_upper_bound(phase,rows)
                require(type(size) is int and size>0,'Available conservative feature byte bound required; no guessed fallback')
                if payload_budget+size+METADATA_RESERVE>CAPACITY:
                    require(updates,'First full raw update exceeds supported window; stop before consumed draw, retain unresolved row')
                    break
                draws=[{**identity,**draw,'phase':phase} for identity,draw in zip(tx.TR_identity,tx.TR_draws)]
                last=tx.TR_draws[-1]
                draws.extend({'source_id':r['source_id'],'language':r['language'],'ordinal':None,'phase':phase,
                    'epoch':last['epoch'],'draw_index':last['draw_index']+1+j} for j,r in enumerate(tx.replay_rows))
                contexts=[{key:d[key] for key in ('source_id','language','epoch','draw_index','phase')}|{'split':'train'} for d in draws]
                entry={'local_update_index':index,'global_update':private.committed_updates()+1,'phase':phase,
                    'batch_id':'update_'+str(private.committed_updates()+1),'TR_draws':deepcopy(tx.TR_draws),
                    'TR_identity':deepcopy(tx.TR_identity),'replay_rows':deepcopy(list(tx.replay_rows)),'rows':deepcopy(rows),
                    'draws':draws,'contexts':contexts,'native_transaction_sha256':tx.identity_digest,
                    'before_sampler_sha256':digest(tx.before),'feature_bytes_upper_bound':size,'prepared_is_committed':False}
                after_replay=deepcopy(tx.after_replay);tr_draws=deepcopy(tx.TR_draws)
            finally:
                tx.abandon()
            private.active_cursor.commit(tr_draws);private.replay.load_state_dict(after_replay)
            entry['after_sampler_sha256']=digest(private.state_dict());updates.append(entry);payload_budget+=size
        require(updates and len(canonical(updates))<=METADATA_RESERVE,'Bounded nonempty complete update metadata required')
        return {'kind':'ASR18_phase_aware_provisional_production_window_plan_v1','production_enabled':False,'phase':phase,
            'finalized_population':finalized_population,'initial_sampler_sha256':digest(before),'final_private_planning_sampler':private.state_dict(),
            'initial_committed_updates':schedule.committed_updates(),'updates':updates,'planned_payload_upper_bound_bytes':payload_budget,
            'metadata_reserve_bytes':METADATA_RESERVE,'capacity_bytes':CAPACITY,'actual_optimizer_updates':0,
            'actual_authoritative_draws_consumed':0,'actual_replay_frontiers_advanced':False,'source_optimizer_or_RNG_reset':False}
    finally:
        require(schedule.state_dict()==before,'Planning touched authoritative cursors/replay/RNG; do not arm')

class WindowCoordinator:
    """One positive Adam then one fsynced journal commit; fail-stop on native errors."""
    def __init__(self,schedule,optimizer,journal,plan,*,resolve_identity,resume=None,verify_resume=None):
        serial();require(plan['kind']=='ASR18_phase_aware_provisional_production_window_plan_v1' and
            plan['capacity_bytes']==CAPACITY and 1<=len(plan['updates'])<=MAX_UPDATES,'Exact bounded phase plan required')
        index=0
        if resume is not None:
            require(type(resume) is dict and set(resume)=={'committed_window_updates','checkpoint','plan_sha256','last_transaction_sha256','sampler_state_sha256','journal_prefix'}
                and type(resume['committed_window_updates']) is int and 1<=resume['committed_window_updates']<=len(plan['updates'])
                and resume['plan_sha256']==digest(plan) and callable(verify_resume),'No bare arbitrary feature/window resume index')
            binding(resume['checkpoint']);index=resume['committed_window_updates'];last=plan['updates'][index-1]
            require(resume['last_transaction_sha256']==last['native_transaction_sha256'] and
                resume['sampler_state_sha256']==last['after_sampler_sha256'] and
                all(journal.binding()[key]==resume['journal_prefix'][key] for key in ('offset','sha256')) and
                verify_resume(deepcopy(resume),schedule.state_dict(),optimizer.updates,journal.binding()) is True,
                'Actual native restore, phase cursors, source RNG and authoritative journal prefix must be verified')
        expected_state=plan['initial_sampler_sha256'] if index==0 else plan['updates'][index-1]['after_sampler_sha256']
        require(not optimizer.poisoned and not schedule.poisoned and
            optimizer.updates==schedule.committed_updates()==plan['initial_committed_updates']+index and
            digest(schedule.state_dict())==expected_state,'Authoritative scientific state differs from provisional window')
        self.schedule=schedule;self.optimizer=optimizer;self.journal=journal;self.plan=deepcopy(plan)
        self.resolve_identity=resolve_identity;self.index=index;self.poisoned=False

    def consume(self,*,prepare_evidence,native_call,verify_preparation):
        serial();require(not self.poisoned and self.index<len(self.plan['updates']),'Consumed/failed window has no next update')
        planned=self.plan['updates'][self.index];before=deepcopy(self.schedule.state_dict());updates=self.optimizer.updates
        tx=self.schedule.propose(optimizer_updates=updates,resolve_TR=self.resolve_identity)
        native_started=False
        try:
            require(tx.identity_digest==planned['native_transaction_sha256'] and digest(before)==planned['before_sampler_sha256']
                and tx.phase==planned['phase'] and tx.TR_draws==planned['TR_draws'] and tx.TR_identity==planned['TR_identity']
                and list(tx.replay_rows)==planned['replay_rows'],'Actual original ordered selection differs from prepared plan')
            evidence=prepare_evidence(planned);binding(evidence)
            require(verify_preparation(planned,evidence) is True and self.schedule.state_dict()==before and self.optimizer.updates==updates,
                    'Actual closed input/targets/context/payload proof required before native work')
            native_started=True;receipt=native_call(planned,tx)
            state=tx.commit(optimizer=self.optimizer,journal=self.journal,native_receipt=receipt,prepared_evidence_binding=evidence)
            require(digest(state)==planned['after_sampler_sha256'],'Committed schedule differs from exact private planning state')
            self.index+=1
            return {'native':receipt,'state':state,'journal':self.journal.binding(),'committed_window_updates':self.index}
        except BaseException:
            if native_started or self.optimizer.updates!=updates or self.optimizer.poisoned or tx.poisoned:
                # Native failure may consume RNG or mutate Adam before raising.
                # Recover only by restoring a prior coherent checkpoint, never
                # by retrying changed live weights with an unadvanced sampler.
                self.poisoned=True;self.optimizer.poisoned=True;self.schedule.poisoned=True
            else:
                require(self.schedule.state_dict()==before,'Failed preparation advanced schedule; stop')
                tx.abandon()
            raise

    def publish_checkpoint(self,*,write_coherent_checkpoint,pointer_path,checkpoint_identity):
        serial();require(not self.poisoned and not self.optimizer.poisoned and not self.schedule.poisoned and self.index>0,
                         'Only actual coherent committed state can publish latest')
        state=self.schedule.state_dict();updates=self.optimizer.updates;journal=self.journal.binding()
        require(updates==self.schedule.committed_updates() and digest(state)==self.plan['updates'][self.index-1]['after_sampler_sha256'],
                'Checkpoint state/frontier differs from last actual commit')
        saved=write_coherent_checkpoint(state,checkpoint_identity)
        binding(saved);metadata=json.loads(Path(saved['path']).read_text())
        require(metadata['kind']=='ASR18_streaming_named_tensor_checkpoint_v1' and metadata['schema_version']==1 and
            metadata['dynamic_module_serialization'] is False and metadata['maximum_CPU_tensor_slice_bytes']<=4*1024**2,
            'Only returned bounded named-state streaming checkpoints may publish')
        binding(metadata['payload'])
        require(self.optimizer.updates==updates and self.schedule.state_dict()==state and self.journal.binding()==journal,
                'Checkpoint writer changed scientific state or journal')
        pointer=Path(pointer_path).resolve();require(pointer.is_relative_to(ROOT.resolve()),'Latest pointer E-only')
        value={'kind':'ASR18_production_window_completed_checkpoint_pointer_v1','checkpoint':saved,'payload':metadata['payload'],
            'committed_updates':updates,'sampler_state_sha256':digest(state),'journal_prefix':journal,'identity':checkpoint_identity,
            'writer_returned_and_payload_verified':True,'committed_window_updates':self.index,
            'all_window_updates_committed':self.index==len(self.plan['updates'])}
        write_json(pointer,value);return value
