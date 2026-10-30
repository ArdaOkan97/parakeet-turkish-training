"""Inactive public integration adapter for EXACTLY TWO new diagnostic pairs.

This intentionally uses the frozen storage2 archive/catalog/publication protocol,
bounded to two rows. It is NOT proof of the separate long-horizon indexed backend.
No existing checkpoint is adopted. The real native writer must create new pairs
at the two pre-reserved paths. Diagnostic flags stay true; winners and unlink are
permanently disabled. A separately exact ROOT source/config review is required.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

import bpe_owned_checkpoint_storage_v2 as frozen
from bpe_broad_audio_v1 import record
from bpe_broad_production_state_v1 import canonical, digest, require

KIND='ASR18_validation_only_diagnostic_owned_checkpoint_pairs_v1'
POLICY='ASR18_reviewed_validation_only_two_new_native_checkpoint_policy_v1'
REVIEW='ROOT_validation_only_two_new_native_checkpoint_storage_review_v1'
TABLE='ASR18_immutable_two_slot_diagnostic_native_writer_path_table_v1'
safe,bound,read,integer=frozen.safe,frozen.bound,frozen.read,frozen.integer


def actual_E_allocation_unit():
    import ctypes
    require(os.name=='nt','This reviewed storage target is Windows E only')
    sectors=ctypes.c_ulong();bytes_per_sector=ctypes.c_ulong();free=ctypes.c_ulong();total=ctypes.c_ulong()
    ok=ctypes.windll.kernel32.GetDiskFreeSpaceW(str(frozen.ROOT.anchor),ctypes.byref(sectors),
        ctypes.byref(bytes_per_sector),ctypes.byref(free),ctypes.byref(total))
    require(ok!=0 and sectors.value>0 and bytes_per_sector.value>0,'Actual E allocation geometry unavailable')
    return sectors.value*bytes_per_sector.value


def policy_values(value):
    require(value['kind']==POLICY and value['accepted'] is True
            and value['scope']=='validation_only_diagnostic' and value['production_enabled'] is False
            and value['production_source_eligible'] is False and value['qualified_winner_eligible'] is False
            and value['existing_checkpoints_adopted_as_owned'] is False
            and value['payload_unlink_enabled'] is False and value['retire_owned_metadata_or_pointer_files'] is False,
            'Permanently diagnostic, unqualified, non-production, new-owned pairs; no deletion authority')
    require(value['global_optimizer_updates']==[21,22] and value['maximum_publications']==2
            and value['hard_live_pair_peak']==5 and value['steady_live_pairs']==4,
            'Exact bounded two-slot21/22 validation; unchanged live/inflight ceiling')
    for key in ('maximum_payload_bytes','maximum_metadata_bytes','maximum_pointer_bytes','maximum_archive_record_bytes',
                'maximum_archive_bytes','maximum_retained_metadata_bytes','maximum_control_bytes',
                'maximum_catalog_bytes','maximum_history_record_bytes','maximum_whole_namespace_allocation_bytes',
                'E_allocation_unit_bytes'):
        integer(value[key],1)
    require(value['maximum_payload_bytes']==256*1024**2 and value['maximum_metadata_bytes']==2*1024**2
            and value['maximum_pointer_bytes']==64*1024 and value['maximum_archive_record_bytes']==1024**2
            and 256<=value['maximum_history_record_bytes']<=8192
            and value['maximum_archive_bytes']>=2*value['maximum_archive_record_bytes']
            and value['maximum_retained_metadata_bytes']>=2*(value['maximum_metadata_bytes']+value['maximum_pointer_bytes'])
            and value['maximum_control_bytes']>=value['maximum_catalog_bytes']+1024**2,
            'Exact declared pair, metadata/archive/control reservations')
    # Frozen complete_publication uses this upper bound only; it does not cause
    # any source-step/Adam mutation or invent the missing first20 updates.
    require(value['planned_native_updates']==22,'Saved actual21/22, not a fresh source restart')
    return value


def verify_diagnostic_pointer(pointer_pin,expectation):
    """Same frozen saved/payload semantics, with honest expected diagnostic=True."""
    require(expectation['storage_validation_scope']=='validation_only_diagnostic'
            and expectation['production_source_eligible'] is False and expectation['qualified_winner_eligible'] is False,
            'No validation result can be a production source/winner')
    pointer=read(pointer_pin);metadata=read(pointer['checkpoint'])
    require(pointer['kind']=='ASR18_production_window_completed_checkpoint_pointer_v2'
            and pointer['writer_returned_and_payload_verified'] is True
            and pointer['all_window_updates_committed'] is True and pointer['diagnostic_only'] is True,
            'Genuine NEW completed diagnostic native writer pointer; never relabel old or diagnostic bytes')
    proof=frozen.verify_saved_metadata(metadata,identity=expectation['identity'],optimizer_updates=expectation['optimizer_updates'],
        sampler_state=expectation['sampler_state'],journal_prefix=expectation['journal_prefix'],diagnostic_only=True)
    require(pointer['identity']==expectation['identity'] and digest(pointer['identity'])==digest(expectation['identity'])
            and type(pointer['committed_updates']) is int and pointer['committed_updates']==expectation['optimizer_updates']
            and pointer['sampler_state_sha256']==digest(expectation['sampler_state'])
            and pointer['journal_prefix']==expectation['journal_prefix']
            and digest(pointer['journal_prefix'])==digest(expectation['journal_prefix'])
            and pointer['saved_semantic_state_verified']==proof and pointer['payload']==metadata['payload'],
            'Exact saved semantic identity/Adam clock/sampler/journal and actual payload agreement')
    require(Path(pointer['checkpoint']['path']).with_suffix('.tensors.bin').resolve()==Path(pointer['payload']['path']).resolve(),
            'Original streamed writer pair suffix retained')
    bound(pointer['payload'])
    return {'pointer':pointer,'metadata':metadata,'expectation':deepcopy(expectation),'semantic_proof':proof}


class ValidationOwnedCheckpointPairs(frozen._StorageProtocol):
    def __init__(self,marker_pin,*,guard):
        self.guard=guard;self.synthetic=False;self.verifier=verify_diagnostic_pointer
        self.marker_pin=deepcopy(marker_pin);self.owner=read(marker_pin);self.folder=safe(self.owner['namespace'])
        require(self.owner['kind']==KIND and self.owner['owner']==frozen.OWNER
                and self.owner['scope']=='validation_only_diagnostic' and self.owner['production_source_eligible'] is False
                and self.owner['qualified_winner_eligible'] is False and self.owner['payload_unlink_enabled'] is False
                and self.owner['component_source']==record(__file__)
                and self.owner['frozen_operation_source']==record(frozen.__file__)
                and Path(marker_pin['path']).resolve()==self.folder/'ownership.json','Exact immutable validation namespace/source/mode')
        self.policy=policy_values(read(self.owner['policy']));review=read(self.owner['ROOT_review'])
        require(review['kind']==REVIEW and review['accepted'] is True and review['reviewer']=='/root'
                and review['component_source']==record(__file__) and review['frozen_operation_source']==record(frozen.__file__)
                and review['policy']==self.owner['policy'] and review['future_namespace']==str(self.folder)
                and review['run_identity_sha256']==digest(self.owner['run_identity'])
                and review['configuration']==self.owner['configuration'] and review['owner_verifier_source']==self.owner['verifier_source']
                and review['protection_plan']==self.owner['protection_plan']
                and review['protected_external']==self.owner['protected_external'],'Exact ROOT validation-only source/config/protection review')
        frozen.protection_plan(self.owner['protection_plan'],self.owner['protected_external'])

    @classmethod
    def create(cls,folder,policy_pin,*,ROOT_review,run_identity,configuration,verifier_source,protected,guard):
        guard();folder=safe(folder);policy=policy_values(read(policy_pin));review=read(ROOT_review)
        require(not folder.exists() and folder.is_relative_to(frozen.ROOT/'runs'),'Fresh E/runs validation namespace only')
        require(review['kind']==REVIEW and review['accepted'] is True and review['reviewer']=='/root'
                and review['policy']==policy_pin and review['component_source']==record(__file__)
                and review['frozen_operation_source']==record(frozen.__file__)
                and review['future_namespace']==str(folder) and review['run_identity_sha256']==digest(run_identity)
                and review['configuration']==configuration and review['owner_verifier_source']==verifier_source,
                'No default approval or caller-selected diagnostic/production mode')
        require(run_identity['storage_scope']=='validation_only_diagnostic'
                and run_identity['production_source_eligible'] is False and run_identity['qualified_winner_eligible'] is False,
                'Immutable scientific/source lineage remains separately diagnostic')
        flattened=frozen.protection_plan(review['protection_plan'],protected)
        require(flattened==review['protected_external'],'Exact immutable original source/history/DEV/winner protections')
        bound(configuration);bound(verifier_source)
        unit=policy['E_allocation_unit_bytes'];require(unit==actual_E_allocation_unit(),'Actual E allocation unit differs from reviewed numeric policy')
        roundup=lambda n:((n+unit-1)//unit)*unit
        minimum=2*sum(roundup(policy[k]) for k in ('maximum_payload_bytes','maximum_metadata_bytes','maximum_pointer_bytes','maximum_archive_record_bytes'))
        minimum+=roundup(policy['maximum_control_bytes'])+3*unit
        require(policy['maximum_whole_namespace_allocation_bytes']>=minimum
                and shutil.disk_usage(frozen.ROOT).free-policy['maximum_whole_namespace_allocation_bytes']>=frozen.RESERVE,
                'BOTH actual writer pairs + archives/control/directories reserved upfront with unchanged E100GB reserve')
        folder.mkdir();(folder/'checkpoints').mkdir();(folder/'archive').mkdir()
        owner={'kind':KIND,'owner':frozen.OWNER,'namespace':str(folder),'nonce':os.urandom(24).hex(),
            'policy':policy_pin,'run_identity':deepcopy(run_identity),'configuration':configuration,
            'component_source':record(__file__),'frozen_operation_source':record(frozen.__file__),
            'verifier_source':verifier_source,'ROOT_review':ROOT_review,'protection_plan':review['protection_plan'],
            'protected_external':flattened,'protected_external_sha256':digest(flattened),
            'scope':'validation_only_diagnostic','production_source_eligible':False,'qualified_winner_eligible':False,
            'payload_unlink_enabled':False,'existing_source_checkpoints_adopted_as_owned':False}
        marker=frozen.new_json(folder/'ownership.json',owner)
        with (folder/'history.jsonl').open('xb') as stream:stream.flush();os.fsync(stream.fileno())
        db=sqlite3.connect(folder/'catalog.sqlite')
        try:db.execute('CREATE TABLE checkpoints (id TEXT PRIMARY KEY, seq INTEGER UNIQUE, value TEXT NOT NULL, retired INTEGER NOT NULL)');db.commit()
        finally:db.close()
        frozen.new_json(folder/'state.json',{'owner':marker,'revision':0,'publications':0,'latest':None,'winners':[],
            'history':{'offset':0,'sha256':hashlib.sha256(b'').hexdigest()},'catalog':record(folder/'catalog.sqlite'),
            'pending_publication':None,'pending_retirement':None,'writer_table':None,'validation_slots':[],
            'validation_completed':0})
        return cls(marker,guard=guard)

    def _load(self):
        # Frozen storage2 _load, with only TWO pre-reserved path slots added to
        # the known namespace. Materialization/hashing is bounded to two pairs;
        # it makes no full-horizon indexed-performance claim.
        self.guard();require(shutil.disk_usage(frozen.ROOT).free>=frozen.RESERVE,'Unchanged E100GB reserve')
        require(read(self.marker_pin)==self.owner and self.owner['protected_external_sha256']==digest(self.owner['protected_external']),'Exact owner/protected list')
        frozen.protection_plan(self.owner['protection_plan'],self.owner['protected_external'])
        state=json.loads((self.folder/'state.json').read_text());require(state['owner']==self.marker_pin,'Exact namespace state')
        require(state['catalog']==record(self.folder/'catalog.sqlite') and state['history']==self._history()
                and state['pending_retirement'] is None,'Unknown catalog/history tail preserved, no automatic reconciliation')
        require(type(state['publications']) is int and type(state['validation_completed']) is int and state['validation_completed']==state['publications']
                and 0<=state['publications']<=2 and state['winners']==[],'No winner/rollback/count drift')
        db=sqlite3.connect(self.folder/'catalog.sqlite')
        try:
            require(db.execute('SELECT count(*) FROM checkpoints').fetchone()[0]<=2,'Bounded table before materialization')
            entries={r[0]:json.loads(r[1])|{'retired':bool(r[2])} for r in db.execute('SELECT id,value,retired FROM checkpoints')}
        finally:db.close()
        require(len(entries)==state['publications'] and state['latest']==(f'p{state["publications"]:09d}' if entries else None),'Exact monotone table publication order')
        known={self.folder/name for name in ('ownership.json','state.json','catalog.sqlite','history.jsonl','operation.lock','checkpoints','archive')}
        pending_paths=set();slots=state['validation_slots']
        if state['writer_table']:
            table=read(state['writer_table']);known.add(safe(state['writer_table']['path'],self.folder))
            require(table['kind']==TABLE and table['slots']==slots and table['ownership']==self.marker_pin
                    and table['configuration']==self.owner['configuration'] and len(slots)==2,'Immutable fixed table/source/config')
            for slot in slots:
                for k in ('metadata_path','payload_path','pointer_path','temporary_path'):pending_paths.add(safe(slot[k],self.folder))
            known|=pending_paths
            expected=slots[state['publications']] if state['publications']<2 else None
            require(state['pending_publication']==expected and digest(state['pending_publication'])==digest(expected),'Exact active slot advances only after actual completion')
        else:require(slots==[] and state['pending_publication'] is None and state['publications']==0,'No unreserved writer path')
        metadata_bytes=archive_bytes=live_bytes=0;registered=set()
        for key,item in entries.items():
            require(item['id']==key and not item['retired'] and 'qualification' not in item,'Validation pairs cannot retire or qualify')
            for field in ('metadata','pointer','payload','archive'):
                pin=item[field];known.add(safe(pin['path'],self.folder));registered.add(Path(pin['path']));bound(pin)
                if field=='archive':archive_bytes+=pin['bytes']
                elif field=='payload':live_bytes+=pin['bytes']
                else:metadata_bytes+=pin['bytes']
        actual={safe(p,self.folder) for p in self.folder.rglob('*')};require(actual<=known,'Unknown file preserved; fixed paths are not general adoption authority')
        for slot in slots:
            for pathkey,cap in (('metadata_path','maximum_metadata_bytes'),('payload_path','maximum_payload_bytes'),('pointer_path','maximum_pointer_bytes'),('temporary_path','maximum_payload_bytes')):
                p=Path(slot[pathkey]);require(not p.exists() or p.stat().st_size<=self.policy[cap],'Actual pending writer cap')
        control=sum(p.stat().st_size for p in actual if p.is_file() and p not in registered and p not in pending_paths)
        require(control<=self.policy['maximum_control_bytes'] and archive_bytes<=self.policy['maximum_archive_bytes']
                and metadata_bytes<=self.policy['maximum_retained_metadata_bytes']
                and (self.folder/'catalog.sqlite').stat().st_size<=self.policy['maximum_catalog_bytes'],'Bounded actual validation namespace capacity')
        for pin in self.owner['protected_external']:bound(pin)
        return state,entries,{'live_payload_bytes':live_bytes,'retained_metadata_bytes':metadata_bytes,'archive_bytes':archive_bytes,'control_bytes':control,'live_pairs':len(entries)}

    def reserve_validation_pair_table(self,*,count=2):
        require(type(count) is int and count==2,'Exactly two slots; no general batch reservation')
        with self.locked():
            state,_,_=self._load();require(state['writer_table'] is None and state['publications']==0,'Reserve both paths ONCE before any native writer')
            require(shutil.disk_usage(frozen.ROOT).free-self.policy['maximum_whole_namespace_allocation_bytes']>=frozen.RESERVE,'Both complete pairs reserved before GPU/native work')
            slots=[]
            for seq,update in enumerate((21,22),1):
                path=self.folder/'checkpoints'/f'p{seq:09d}.json'
                slot={'kind':'ASR18_owned_future_checkpoint_publication_intent_v1','ownership':self.marker_pin,
                    'id':f'p{seq:09d}','seq':seq,'expected_optimizer_updates':update,
                    'metadata_path':str(path),'payload_path':str(path.with_suffix('.tensors.bin')),
                    'temporary_path':str(path.with_suffix('.tensors.writing.bin')),'pointer_path':str(path.with_suffix('.pointer.json')),
                    'maximum_additional_owned_bytes':sum(self.policy[k] for k in ('maximum_payload_bytes','maximum_metadata_bytes','maximum_pointer_bytes','maximum_archive_record_bytes')),
                    'storage_scope':'validation_only_diagnostic','production_source_eligible':False}
                require(not any(Path(slot[k]).exists() for k in ('metadata_path','payload_path','temporary_path','pointer_path')),'Fresh exact paths; never overwrite/adopt')
                slots.append(slot)
            table={'kind':TABLE,'ownership':self.marker_pin,'configuration':self.owner['configuration'],
                'component_source':record(__file__),'frozen_operation_source':record(frozen.__file__),
                'scope':'validation_only_diagnostic','production_source_eligible':False,'qualified_winner_eligible':False,
                'slots':slots,'both_pair_capacities_reserved_before_any_native_writer':True}
            pin=frozen.new_json(self.folder/'writer_paths.json',table)
            state.update(writer_table=pin,validation_slots=slots,pending_publication=deepcopy(slots[0]))
            self._save(state,{'event':'exact_two_new_diagnostic_writer_slots_reserved_before_native','table':pin,'global_optimizer_updates':[21,22]})
            return {'table':pin,'slots':deepcopy(slots),'ownership':self.marker_pin}

    def _save(self,state,event):
        if event['event']=='verified_owned_checkpoint_published':
            require(state['validation_completed']<2 and event['id']==state['validation_slots'][state['validation_completed']]['id'],'Exact native completion order21 then22')
            state['validation_completed']+=1
            state['pending_publication']=deepcopy(state['validation_slots'][state['validation_completed']]) if state['validation_completed']<2 else None
        return super()._save(state,event)

    def complete_validation_publication(self,intent,pointer_pin,expectation,proof_pin,verify_current_owner):
        require(type(expectation['optimizer_updates']) is int and expectation['optimizer_updates']==intent['expected_optimizer_updates']
                and expectation['storage_validation_scope']=='validation_only_diagnostic'
                and expectation['production_source_eligible'] is False and expectation['qualified_winner_eligible'] is False,
                'Exact21/22 native saved expectation, never production promotion')
        proof=read(proof_pin)
        require(proof['validation_writer_intent']==intent and proof['native_PRE_and_saved_persistence_verified'] is True
                and proof['actual_native_coherent_idle_or_exit_verified'] is True
                and proof['actual_new_checkpoint_pointer']==pointer_pin,'Source-bound actual PRE/SAVED/persistence/coherent owner, not metadata-only success')
        return super().complete_publication(intent,pointer_pin,expectation,proof_pin,verify_current_owner)

    def retirement_admission_dry_run(self,checkpoint_ids,proof_pin,verify_current_owner):
        require(type(checkpoint_ids) is list and checkpoint_ids==['p000000001'],'Only obsolete NEW21 after NEW22, never old20 or active latest')
        with self.locked():
            state,entries,_=self._load();require(state['publications']==2 and state['pending_publication'] is None,'Both actual publications required')
            proof=self._verify_owner(proof_pin,'validation_retirement_ADMISSION_ONLY_no_unlink',state,verify_current_owner)
            require(proof['retire_checkpoint_ids']==checkpoint_ids and proof['all_new_validation_captures_fresh_recovered0'] is True,
                    'Every new21/22 actual recovery0 before optional admission-only check')
            item=entries[checkpoint_ids[0]]
            for field in ('metadata','pointer','payload','archive'):bound(item[field])
            self._journal(item['journal_prefix'])
            return {'kind':'ASR18_validation_checkpoint_retirement_admission_dry_run_v1','passed':True,
                'ownership':self.marker_pin,'targets':checkpoint_ids,'actual_unlink_performed':False,
                'all_metadata_payloads_and_archives_preserved':True,'production_source_eligible':False}

    def reserve_publication(self,*args,**kwargs):raise ValueError('Use the exactly-two validation table, never normal production reservation')
    def complete_publication(self,*args,**kwargs):raise ValueError('Use the diagnostic-specific verified completion API')
    def protect_qualified_winner(self,*args,**kwargs):raise ValueError('Validation namespaces are permanently unqualified/non-production')
    def retire_obsolete_payloads(self,*args,**kwargs):raise ValueError('No actual unlink authority in this validation-only component')
