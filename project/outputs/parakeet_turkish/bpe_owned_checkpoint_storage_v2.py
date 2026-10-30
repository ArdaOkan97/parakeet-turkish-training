"""Inactive bounded storage for a separately reviewed FUTURE run namespace.

Only explicitly registered obsolete future-owned tensor payloads can retire.
Metadata, pointer, source/DEV/winner links and authoritative journal prefixes
remain in immutable archives/history. No model/runtime import or launch CLI.
"""
from contextlib import contextmanager
from copy import deepcopy
import gzip, hashlib, inspect, json, os, re, shutil, sqlite3, stat, time
from pathlib import Path

from bpe_broad_production_state_v1 import require, canonical, digest
from bpe_broad_audio_v1 import record
from bpe_streaming_checkpoint_semantics_v2 import verify_saved_metadata

ROOT=Path(r'E:\gpu_training_scratchpad')
POLICY_KIND='ASR18_reviewed_future_owned_checkpoint_storage_policy_v1'
OWNER_KIND='ASR18_future_owned_checkpoint_storage_namespace_v1'
OWNER='/root/memory_diagnosis'
RESERVE=100_000_000_000


def integer(value, low=0):
    require(type(value) is int and value>=low,'Exact nonboolean storage count/byte bound');return value


def safe(path, namespace=None):
    path=Path(path).absolute()
    require(path.is_relative_to(ROOT.absolute()),'Future owned storage E-only; no original project path')
    current=path
    while current.is_relative_to(ROOT.absolute()):
        if current.exists():
            attributes=os.lstat(current).st_file_attributes if os.name=='nt' else 0
            require(not attributes & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024) and not current.is_symlink(),
                    'No symbolic/reparse redirection in owned path')
        if current==ROOT.absolute():break
        current=current.parent
    resolved=path.resolve();require(resolved.is_relative_to(ROOT.resolve()),'Resolved owned storage escaped E')
    if namespace is not None:require(resolved.is_relative_to(Path(namespace).resolve()),'Exact owned namespace containment')
    return resolved


def bound(pin, *, maximum=None):
    require(type(pin) is dict and set(pin)=={'path','bytes','sha256'} and integer(pin['bytes'],1)
            and (maximum is None or pin['bytes']<=maximum) and record(pin['path'])==pin,'Exact immutable bound file required')
    return pin


def read(pin, maximum=16*1024**2):
    bound(pin,maximum=maximum);return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))


def new_json(path, value):
    path=safe(path);raw=canonical(value)
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return record(path)


def replace_json(path, value):
    path=safe(path);temporary=path.with_suffix('.pending-state.json')
    require(not temporary.exists(),'Unreconciled state transaction remains; no automatic retry')
    new_json(temporary,value);os.replace(temporary,path)


def verify_streamed_pointer(pointer_pin, expectation):
    """Reuse the frozen scalar semantic verifier; do not reconstruct model state."""
    pointer=read(pointer_pin);metadata=read(pointer['checkpoint'])
    require(pointer['kind']=='ASR18_production_window_completed_checkpoint_pointer_v2'
            and pointer['writer_returned_and_payload_verified'] is True
            and pointer['all_window_updates_committed'] is True and pointer['diagnostic_only'] is False,
            'Actual full-run coherent latest pointer; no finite diagnostic promotion')
    proof=verify_saved_metadata(metadata,identity=expectation['identity'],optimizer_updates=expectation['optimizer_updates'],
        sampler_state=expectation['sampler_state'],journal_prefix=expectation['journal_prefix'],diagnostic_only=False)
    require(pointer['identity']==expectation['identity'] and digest(pointer['identity'])==digest(expectation['identity'])
            and pointer['committed_updates']==expectation['optimizer_updates'] and type(pointer['committed_updates']) is int
            and pointer['sampler_state_sha256']==digest(expectation['sampler_state'])
            and pointer['journal_prefix']==expectation['journal_prefix']
            and digest(pointer['journal_prefix'])==digest(expectation['journal_prefix'])
            and pointer['saved_semantic_state_verified']==proof and pointer['payload']==metadata['payload'],
            'Live expected snapshot, scalar saved state and published semantic identity disagree')
    target=Path(pointer['checkpoint']['path']).with_suffix('.tensors.bin').resolve()
    require(target==Path(pointer['payload']['path']).resolve(),'Exact original writer pair suffix rule')
    bound(pointer['payload'])
    return {'pointer':pointer,'metadata':metadata,'expectation':deepcopy(expectation),'semantic_proof':proof}


def policy_values(policy, *, synthetic=False):
    require(policy['kind']==('ASR18_synthetic_owned_checkpoint_policy_v1' if synthetic else POLICY_KIND), 'Narrow policy kind')
    require(policy['production_enabled'] is False and policy['retire_existing_or_original_artifacts'] is False
            and policy['retire_owned_metadata_or_pointer_files'] is False
            and policy['publication_cadence']=='at_most_one_payload_per_actual_positive_optimizer_update',
            'No old/original/metadata deletion authority or unlimited duplicate publication cadence')
    for key in ('maximum_payload_bytes','maximum_metadata_bytes','maximum_pointer_bytes','maximum_archive_record_bytes',
                'maximum_publications','maximum_archive_bytes','maximum_retained_metadata_bytes','maximum_control_bytes',
                'maximum_history_record_bytes','maximum_catalog_bytes'):
        integer(policy[key],1)
    require(integer(policy['steady_live_pairs'],1)==4 and integer(policy['maximum_inflight_publications'],1)==1
            and integer(policy['hard_live_pair_peak'],1)==5,'Exact latest+up to3winner and one in-flight pair bounds')
    require(policy['maximum_publications']>=integer(policy['planned_native_updates'],1)
            and policy['maximum_archive_bytes']>=(policy['maximum_publications']+3)*policy['maximum_archive_record_bytes']
            and policy['maximum_retained_metadata_bytes']>=policy['maximum_publications']*(policy['maximum_metadata_bytes']+policy['maximum_pointer_bytes'])
            and 256<=policy['maximum_history_record_bytes']<=8192
            and policy['maximum_control_bytes']>=policy['maximum_publications']*3*policy['maximum_history_record_bytes']+policy['maximum_catalog_bytes']+1024**2,
            'Worst geometry-shortened windows/publication count and all retained metadata/archive bytes must fit explicit caps')
    return policy


def protection_plan(plan_pin, protected):
    """No free caller list may inherit an approval for different protections."""
    plan=read(plan_pin)
    require(plan['kind']=='ASR18_future_owned_checkpoint_protection_plan_v1'
            and plan['adopt_external_sources_as_owned'] is False
            and set(plan['required_roles'])=={'source','history','DEV','qualified_winner'},
            'Exact required source/history/DEV/winner protection plan')
    flattened=[];seen={}
    for role in ('source','history','DEV','qualified_winner'):
        pins=plan['required_roles'][role]
        require(type(pins) is list and 1<=len(pins)<=64,'Every reviewed protected role must be nonempty/bounded')
        for pin in pins:
            bound(pin)
            if pin['path'] in seen:require(digest(seen[pin['path']])==digest(pin),'Conflicting protected file identity')
            else:seen[pin['path']]=pin;flattened.append(deepcopy(pin))
    require(type(protected) is list and protected and protected==flattened and digest(protected)==digest(flattened),
            'Omitted, changed, reordered or empty caller protection list cannot use the same review')
    return flattened


class _StorageProtocol:
    """Owned file engine. Synthetic tests use a distinct non-public owner kind."""
    def __init__(self, marker_pin, *, checkpoint_verifier, guard, synthetic=False):
        self.guard=guard;self.verifier=checkpoint_verifier;self.synthetic=synthetic
        self.owner=read(marker_pin);self.marker_pin=deepcopy(marker_pin);self.folder=safe(self.owner['namespace'])
        require(self.owner['kind']==('ASR18_synthetic_owned_checkpoint_namespace_v1' if synthetic else OWNER_KIND)
                and Path(marker_pin['path']).resolve()==self.folder/'ownership.json'
                and self.owner['owner']==OWNER and self.owner['component_source']==record(__file__), 'Exact owner namespace/source marker')
        self.policy=policy_values(read(self.owner['policy']),synthetic=synthetic)

    @classmethod
    def initialize(cls, folder, policy_pin, *, run_identity, verifier_source, protected, checkpoint_verifier, guard, synthetic=False, ROOT_review=None, protection_plan_binding=None):
        guard();folder=safe(folder);require(not folder.exists(),'Fresh future namespace only; never adopt original/current files')
        policy=policy_values(read(policy_pin),synthetic=synthetic)
        # Before a production namespace exists, reserve the entire explicitly
        # bounded history horizon as well as peak live/in-flight payloads.
        required=policy['hard_live_pair_peak']*policy['maximum_payload_bytes']+policy['maximum_archive_bytes']+policy['maximum_retained_metadata_bytes']+policy['maximum_control_bytes']
        require(shutil.disk_usage(ROOT).free-required>=RESERVE,'Full declared bounded storage horizon violates E100GB reserve')
        folder.mkdir();(folder/'checkpoints').mkdir();(folder/'archive').mkdir()
        owner={'kind':'ASR18_synthetic_owned_checkpoint_namespace_v1' if synthetic else OWNER_KIND,'owner':OWNER,
            'namespace':str(folder),'nonce':os.urandom(24).hex(),'policy':policy_pin,'run_identity':deepcopy(run_identity),
            'component_source':record(__file__),'verifier_source':verifier_source,'protected_external':deepcopy(protected),'ROOT_review':ROOT_review,
            'protection_plan':protection_plan_binding,'protected_external_sha256':digest(protected),
            'existing_source_checkpoints_adopted_as_owned':False}
        marker=new_json(folder/'ownership.json',owner)
        with (folder/'history.jsonl').open('xb') as stream:stream.flush();os.fsync(stream.fileno())
        db=sqlite3.connect(folder/'catalog.sqlite')
        try:
            db.execute('CREATE TABLE checkpoints (id TEXT PRIMARY KEY, seq INTEGER UNIQUE, value TEXT NOT NULL, retired INTEGER NOT NULL)')
            db.commit()
        finally:db.close()
        state={'owner':marker,'revision':0,'publications':0,'latest':None,'winners':[],
            'history':{'offset':0,'sha256':hashlib.sha256(b'').hexdigest()},'catalog':record(folder/'catalog.sqlite'),
            'pending_publication':None,'pending_retirement':None}
        new_json(folder/'state.json',state)
        return cls(marker,checkpoint_verifier=checkpoint_verifier,guard=guard,synthetic=synthetic)

    @contextmanager
    def locked(self):
        self.guard();path=safe(self.folder/'operation.lock',self.folder)
        pin=new_json(path,{'owner':self.marker_pin,'PID':os.getpid(),'nonce':os.urandom(16).hex()})
        try:yield
        finally:
            require(record(path)==pin,'Owned operation lock changed; never unlink unknown lock')
            path.unlink()

    def _history(self):
        sha=hashlib.sha256();size=0
        with (self.folder/'history.jsonl').open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):self.guard();sha.update(block);size+=len(block)
        return {'offset':size,'sha256':sha.hexdigest()}

    def _load(self):
        self.guard();require(shutil.disk_usage(ROOT).free>=RESERVE,'Unchanged E100GB reserve required at storage boundary')
        require(read(self.marker_pin)==self.owner,'Ownership marker changed')
        require(self.owner['protected_external_sha256']==digest(self.owner['protected_external']), 'Owner protected-list digest differs')
        if not self.synthetic:protection_plan(self.owner['protection_plan'],self.owner['protected_external'])
        state=json.loads((self.folder/'state.json').read_text());require(state['owner']==self.marker_pin,'Owned state belongs to another namespace')
        require(state['catalog']==record(self.folder/'catalog.sqlite') and state['history']==self._history(),
                'Unknown catalog/history tail or stale metadata; explicit reconciliation required')
        require(state['pending_retirement'] is None,'Incomplete retirement must be reconciled by native owner, not retried')
        db=sqlite3.connect(self.folder/'catalog.sqlite')
        try:entries={row[0]:json.loads(row[1])|{'retired':bool(row[2])} for row in db.execute('SELECT id,value,retired FROM checkpoints')}
        finally:db.close()
        require(len(entries)==state['publications'],'Catalog/publication count mismatch')
        require(state['latest']==(max(entries.values(),key=lambda v:v['seq'])['id'] if entries else None), 'Latest pointer cannot roll back to an older valid checkpoint')
        require(len(set(state['winners']))==len(state['winners']) and set(state['winners'])=={key for key,value in entries.items() if 'qualification' in value},
                'Qualified winner protection cannot silently disappear from catalog/state')
        known={self.folder/'ownership.json',self.folder/'state.json',self.folder/'catalog.sqlite',self.folder/'history.jsonl',self.folder/'operation.lock',self.folder/'checkpoints',self.folder/'archive'}
        metadata_bytes=archive_bytes=live_bytes=0
        for value in entries.values():
            for key in ('metadata','pointer','archive',*(['qualification'] if 'qualification' in value else [])):
                pin=value[key];path=safe(pin['path'],self.folder);known.add(path);bound(pin)
                if key in ('archive','qualification'):archive_bytes+=pin['bytes']
                else:metadata_bytes+=pin['bytes']
            payload=value['payload'];path=safe(payload['path'],self.folder)
            if value['retired']:require(not path.exists(),'Retired payload reappeared; unknown restore conflict')
            else:known.add(path);bound(payload);live_bytes+=payload['bytes']
        pending=state['pending_publication']
        if pending:
            for key in ('metadata_path','payload_path','pointer_path','temporary_path'):known.add(safe(pending[key],self.folder))
        actual={safe(p,self.folder) for p in self.folder.rglob('*')}
        require(actual<=known,'Unknown file/directory in strictly owned namespace; no cleanup authority')
        require(archive_bytes<=self.policy['maximum_archive_bytes'] and metadata_bytes<=self.policy['maximum_retained_metadata_bytes'], 'Bounded history/archive capacity exceeded')
        control=sum(p.stat().st_size for p in actual if p.is_file() and p not in {Path(v[k]['path']) for v in entries.values() for k in ('metadata','pointer','archive','payload',*(['qualification'] if 'qualification' in v else []))})
        require(control<=self.policy['maximum_control_bytes'] and (self.folder/'catalog.sqlite').stat().st_size<=self.policy['maximum_catalog_bytes'],
                'Explicit control/catalog/history cap exceeded')
        for protected in self.owner['protected_external']:bound(protected)
        live=[v for v in entries.values() if not v['retired']]
        require(len(live)<=self.policy['hard_live_pair_peak'],'Live owned pair cap exceeded')
        return state,entries,{'live_payload_bytes':live_bytes,'retained_metadata_bytes':metadata_bytes,'archive_bytes':archive_bytes,'control_bytes':control,'live_pairs':len(live)}

    def _save(self,state,event):
        raw=canonical(event)+b'\n';require(len(raw)<=self.policy['maximum_history_record_bytes'],'Small bounded storage history event')
        control=(self.folder/'catalog.sqlite').stat().st_size+(self.folder/'history.jsonl').stat().st_size+len(raw)+len(canonical(state))+16384
        require(control<=self.policy['maximum_control_bytes'] and (self.folder/'catalog.sqlite').stat().st_size<=self.policy['maximum_catalog_bytes'],
                'Prospective control/history/catalog cap exceeded; preserve incomplete transaction')
        with (self.folder/'history.jsonl').open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        state['history']=self._history();state['catalog']=record(self.folder/'catalog.sqlite');state['revision']+=1
        replace_json(self.folder/'state.json',state)

    def _verify_owner(self, proof_pin, operation, state, verify_current_owner):
        proof=read(proof_pin)
        db=sqlite3.connect(self.folder/'catalog.sqlite')
        try:latest=json.loads(db.execute('SELECT value FROM checkpoints WHERE id=?',(state['latest'],)).fetchone()[0])['pointer'] if state['latest'] else None
        finally:db.close()
        require(proof['kind']=='ASR18_owned_checkpoint_mapping_release_quiescence_v1'
            and proof['owner']==OWNER and proof['ownership']==self.marker_pin
            and proof['state_revision']==state['revision'] and proof['catalog']==state['catalog']
            and proof['coherent_latest_pointer']==latest and proof['protected_owned_winners']==state['winners']
            and type(proof['nonce']) is str and len(proof['nonce'])>=32
            and all(proof[k] is True for k in ('native_work_quiescent','checkpoint_mappings_released','candidate_payload_handles_released'))
            and proof['checkpoint_write_inflight'] is False and proof['restore_in_progress'] is False,
            'Distinct exact native-owner mapping-release attestation required; idle or CLOSED0 alone is insufficient')
        bound(self.owner['verifier_source'])
        require(callable(verify_current_owner) and str(Path(inspect.getsourcefile(verify_current_owner)).resolve())==self.owner['verifier_source']['path']
            and verify_current_owner({'proof':proof_pin,'operation':operation,'ownership':self.marker_pin,'state_revision':state['revision'],
                                     'latest':state['latest'],'winners':state['winners']}) is True,
            'Current owner/lease/nonce/mapping-release/restore-exclusion witness must verify at actual operation')
        return proof

    def _journal(self,prefix):
        path=safe(prefix['path']);require(not path.is_relative_to(self.folder/'checkpoints'), 'Authoritative journal is never an owned tensor payload')
        remaining=integer(prefix['offset']);require(path.stat().st_size>=remaining,'Authoritative journal prefix truncated')
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            while remaining:
                self.guard();block=stream.read(min(1024**2,remaining));require(block,'Full authoritative prefix required');sha.update(block);remaining-=len(block)
        require(sha.hexdigest()==prefix['sha256'],'Authoritative journal prefix/hash differs')

    def _archive(self,path,value):
        require(shutil.disk_usage(ROOT).free-self.policy['maximum_archive_record_bytes']>=RESERVE,'Prospective immutable archive violates E100GB reserve')
        with path.open('xb') as stream:
            with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as zipped:
                for part in json.JSONEncoder(sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).iterencode(value):zipped.write(part.encode())
            stream.flush();os.fsync(stream.fileno())
        pin=record(path);require(pin['bytes']<=self.policy['maximum_archive_record_bytes'] and shutil.disk_usage(ROOT).free>=RESERVE,
                                 'Compressed immutable archive cap/E100GB exceeded; preserve all payloads unresolved')
        return pin

    def reserve_publication(self):
        with self.locked():
            state,entries,usage=self._load();policy=self.policy
            require(state['pending_publication'] is None and usage['live_pairs']<=policy['steady_live_pairs'],
                    'Retire exact obsolete pair before next bounded publication; no automatic deletion')
            require(state['publications']<policy['maximum_publications'],'Declared worst-case publication count exhausted before write')
            extra=policy['maximum_payload_bytes']+policy['maximum_metadata_bytes']+policy['maximum_pointer_bytes']+policy['maximum_archive_record_bytes']+16384
            require(usage['archive_bytes']+policy['maximum_archive_record_bytes']<=policy['maximum_archive_bytes']
                and usage['retained_metadata_bytes']+policy['maximum_metadata_bytes']+policy['maximum_pointer_bytes']<=policy['maximum_retained_metadata_bytes']
                and shutil.disk_usage(ROOT).free-extra>=RESERVE,'Prospective publication/history capacity or E100GB admission denied before writer')
            number=state['publications']+1;name=f'p{number:09d}';metadata=self.folder/'checkpoints'/(name+'.json')
            pending={'kind':'ASR18_owned_future_checkpoint_publication_intent_v1','ownership':self.marker_pin,'id':name,'seq':number,
                'metadata_path':str(metadata),'payload_path':str(metadata.with_suffix('.tensors.bin')),
                'temporary_path':str(metadata.with_suffix('.tensors.writing.bin')),
                'pointer_path':str(metadata.with_suffix('.pointer.json')),'maximum_additional_owned_bytes':extra}
            require(not any(Path(pending[k]).exists() for k in ('metadata_path','payload_path','temporary_path','pointer_path')),'Never overwrite an existing checkpoint pair')
            state['pending_publication']=pending;self._save(state,{'event':'publication_reserved_before_write','id':name,'maximum_additional_owned_bytes':extra})
            return deepcopy(pending)

    def complete_publication(self,intent,pointer_pin,expectation,proof_pin,verify_current_owner):
        with self.locked():
            state,entries,usage=self._load();require(state['pending_publication']==intent and digest(state['pending_publication'])==digest(intent),'Exact current publication intent required')
            proof=self._verify_owner(proof_pin,'complete_publication',state,verify_current_owner)
            require(not Path(intent['temporary_path']).exists(),'Writer temporary file remains; no completion from metadata alone')
            require(pointer_pin['path']==intent['pointer_path'],'Exact owned fresh pointer path')
            verified=self.verifier(pointer_pin,expectation);pointer=verified['pointer'];metadata=verified['metadata']
            require(pointer['checkpoint']['path']==intent['metadata_path'] and pointer['payload']['path']==intent['payload_path']
                and pointer['checkpoint']['bytes']<=self.policy['maximum_metadata_bytes']
                and pointer_pin['bytes']<=self.policy['maximum_pointer_bytes'] and pointer['payload']['bytes']<=self.policy['maximum_payload_bytes'],
                'Verified writer pair exceeds exact preallocation or belongs to another namespace')
            updates=integer(expectation['optimizer_updates'],1)
            require(updates<=self.policy['planned_native_updates'] and (not state['latest'] or updates>entries[state['latest']]['optimizer_updates']),
                    'Stale/duplicate payload publication cannot become latest')
            require(expectation['run_identity']==self.owner['run_identity'] and digest(expectation['run_identity'])==digest(self.owner['run_identity']), 'Original full-run scientific source identity must match owner')
            self._journal(expectation['journal_prefix'])
            archive={'kind':'ASR18_future_owned_checkpoint_immutable_metadata_archive_v1','ownership':self.marker_pin,
                'id':intent['id'],'metadata':metadata,'metadata_binding':pointer['checkpoint'],'pointer':pointer,'pointer_binding':pointer_pin,
                'actual_live_expectation':deepcopy(expectation),'owner_mapping_release':proof,'owner_mapping_release_binding':proof_pin,
                'protected_external_source_history_DEV_and_winner':self.owner['protected_external'],
                'previous_latest':state['latest'],'qualified_owned_winners':deepcopy(state['winners']),'authoritative_history_prefix_before_publication':state['history']}
            target=self.folder/'archive'/(intent['id']+'.json.gz')
            archived=self._archive(target,archive)
            item={'id':intent['id'],'seq':intent['seq'],'metadata':pointer['checkpoint'],'pointer':pointer_pin,'payload':pointer['payload'],
                  'archive':archived,'optimizer_updates':updates,'journal_prefix':deepcopy(expectation['journal_prefix']),'run_identity_sha256':digest(expectation['run_identity'])}
            self._verify_owner(proof_pin,'publish_latest_after_fsynced_archive',state,verify_current_owner)
            db=sqlite3.connect(self.folder/'catalog.sqlite')
            try:db.execute('INSERT INTO checkpoints VALUES (?,?,?,0)',(item['id'],item['seq'],canonical(item).decode()));db.commit()
            finally:db.close()
            state['latest']=item['id'];state['publications']+=1;state['pending_publication']=None
            self._save(state,{'event':'verified_owned_checkpoint_published','id':item['id'],'archive':archived,'optimizer_updates':updates})
            return deepcopy(item)

    def protect_qualified_winner(self,checkpoint_id,proof_pin,verify_current_owner):
        with self.locked():
            state,entries,_=self._load();require(checkpoint_id in entries and not entries[checkpoint_id]['retired'],'Only an existing owned coherent pair may be newly protected')
            proof=self._verify_owner(proof_pin,'protect_qualified_winner',state,verify_current_owner)
            require(proof['strict_all_six_qualified'] is True and proof['qualified_pointer']==entries[checkpoint_id]['pointer'], 'Native owner must verify actual official all6 qualification')
            require(checkpoint_id in state['winners'] or len(state['winners'])<3,'Only3 declared full-run DEV boundary winners; never evict protected source/history')
            require(checkpoint_id not in state['winners'],'Immutable qualification already registered; no duplicate publication')
            qualified=self._archive(self.folder/'archive'/(checkpoint_id+'.qualified.json.gz'),
                {'kind':'ASR18_immutable_owned_checkpoint_qualification_archive_v1','ownership':self.marker_pin,
                 'pointer':entries[checkpoint_id]['pointer'],'native_owner_qualification_proof':proof,
                 'proof_binding':proof_pin,'source_history_links':self.owner['protected_external']})
            item=deepcopy(entries[checkpoint_id]);item.pop('retired');item['qualification']=qualified
            db=sqlite3.connect(self.folder/'catalog.sqlite')
            try:db.execute('UPDATE checkpoints SET value=? WHERE id=?',(canonical(item).decode(),checkpoint_id));db.commit()
            finally:db.close()
            state['winners'].append(checkpoint_id)
            self._save(state,{'event':'owned_qualified_winner_protected','id':checkpoint_id,'proof':proof_pin})

    def retire_obsolete_payloads(self,checkpoint_ids,proof_pin,verify_current_owner):
        require(type(checkpoint_ids) is list and checkpoint_ids and len(set(checkpoint_ids))==len(checkpoint_ids),'Explicit unique obsolete owned IDs only')
        with self.locked():
            state,entries,_=self._load();require(state['pending_publication'] is None,'No retirement during incomplete checkpoint publication')
            proof=self._verify_owner(proof_pin,'retire_obsolete_payloads',state,verify_current_owner)
            require(proof['retire_checkpoint_ids']==checkpoint_ids,'Attestation must name exact ordered candidate IDs')
            candidates=[]
            protected={str(Path(p['path']).resolve()) for p in self.owner['protected_external']}
            for key in checkpoint_ids:
                require(key in entries and key!=state['latest'] and key not in state['winners'] and not entries[key]['retired'],'Unknown/latest/qualified/retired pair may not be removed')
                item=entries[key];payload=safe(item['payload']['path'],self.folder)
                require(payload==Path(item['metadata']['path']).with_suffix('.tensors.bin').resolve() and str(payload) not in protected,'Only exact future-owned obsolete payload, never external source/winner')
                for field in ('metadata','pointer','payload','archive'):bound(item[field])
                self._journal(item['journal_prefix'])
                candidates.append(item)
            # Archive already includes complete saved state/source/prefix links.
            # Record a durable exact retirement intent BEFORE the first unlink.
            state['pending_retirement']={'checkpoint_ids':checkpoint_ids,'proof':proof_pin,'archived_metadata':[v['archive'] for v in candidates]}
            replace_json(self.folder/'state.json',state)
            for item in candidates:
                self.guard();self._verify_owner(proof_pin,'unlink_exact_archived_obsolete_payload_'+item['id'],state,verify_current_owner)
                path=safe(item['payload']['path'],self.folder);bound(item['payload']);path.unlink()
                require(not path.exists(),'Actual owned payload retirement not observed')
            db=sqlite3.connect(self.folder/'catalog.sqlite')
            try:
                for key in checkpoint_ids:db.execute('UPDATE checkpoints SET retired=1 WHERE id=?',(key,))
                db.commit()
            finally:db.close()
            state['pending_retirement']=None
            self._save(state,{'event':'exact_owned_obsolete_payloads_retired','ids':checkpoint_ids,'proof':proof_pin})
            return {'passed':True,'retired_owned_payloads':[v['payload'] for v in candidates],
                    'preserved_metadata_pointer_and_archive':[v['archive'] for v in candidates],
                    'original_or_current_checkpoint_changed':False,'latest':state['latest'],'qualified_winners':state['winners']}


class OwnedFutureCheckpointStore(_StorageProtocol):
    """Public gate: no current namespace adoption, synthetic proof or proposal arm."""
    def __init__(self, marker_pin, *, checkpoint_verifier=verify_streamed_pointer, guard, synthetic=False):
        require(synthetic is False and checkpoint_verifier is verify_streamed_pointer,'Public store always uses frozen real semantic verifier')
        super().__init__(marker_pin,checkpoint_verifier=verify_streamed_pointer,guard=guard,synthetic=False)
        policy=read(self.owner['policy']);review=read(self.owner['ROOT_review'])
        require(policy['accepted'] is True and policy['full_horizon_accounting_reviewed'] is True
                and review['kind']=='ASR18_ROOT_future_owned_checkpoint_storage_review_v1' and review['reviewer']=='/root'
                and review['accepted'] is True and review['policy']==self.owner['policy']
                and review['component_source']==record(__file__) and review['future_namespace']==str(self.folder)
                and review['run_identity_sha256']==digest(self.owner['run_identity'])
                and review['owner_verifier_source']==self.owner['verifier_source']
                and review['protection_plan']==self.owner['protection_plan']
                and review['protected_external']==self.owner['protected_external']
                and digest(review['protected_external'])==self.owner['protected_external_sha256'],
                'Existing public handle still requires exact ROOT future-policy/protection review')
        protection_plan(review['protection_plan'],self.owner['protected_external'])
    @classmethod
    def create(cls,folder,policy_pin,*,ROOT_review,run_identity,verifier_source,protected,guard):
        policy=read(policy_pin);review=read(ROOT_review)
        require(policy['kind']==POLICY_KIND and policy['accepted'] is True and policy['full_horizon_accounting_reviewed'] is True
                and review['kind']=='ASR18_ROOT_future_owned_checkpoint_storage_review_v1' and review['reviewer']=='/root'
                and review['accepted'] is True and review['policy']==policy_pin and review['component_source']==record(__file__)
                and review['future_namespace']==str(safe(folder)) and review['run_identity_sha256']==digest(run_identity)
                and review['owner_verifier_source']==verifier_source,'Exact separately reviewed FUTURE namespace/policy/source/owner verifier required')
        flattened=protection_plan(review['protection_plan'],protected)
        require(review['protected_external']==flattened and digest(review['protected_external'])==digest(flattened),
                'ROOT approval must bind the exact complete required protected list')
        require(safe(folder).is_relative_to(ROOT/'runs'),'Production owned namespace must be fresh E/runs, never analysis/current/original storage')
        for pin in protected:bound(pin)
        bound(verifier_source)
        return cls.initialize(folder,policy_pin,run_identity=run_identity,verifier_source=verifier_source,protected=protected,
                              checkpoint_verifier=verify_streamed_pointer,guard=guard,synthetic=False,ROOT_review=ROOT_review,
                              protection_plan_binding=review['protection_plan'])
