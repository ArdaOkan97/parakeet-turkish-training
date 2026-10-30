"""Explicit operational verifier successor over unchanged indexed storage.

Creator ownership, quotas, schema, publication and retirement algorithms remain
unchanged. The actual new callback source is ROOT pinned independently of the
legacy creator's callback contract. No existing owner or policy is rewritten.
"""
from copy import deepcopy
import inspect,json,sqlite3
from pathlib import Path
from asr18_BPE9223_indexed_checkpoint_storage_v1 import IndexedOwnedFutureCheckpointStore as Original
from asr18_BPE9223_indexed_checkpoint_storage_v1 import bound,read,safe,digest,require
from asr18_BPE9223_owned_checkpoint_storage_v1 import OWNER
from bpe_broad_audio_v1 import record
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility

class IndexedOwnedFutureCheckpointStore(Original):
    def __init__(self, marker_pin, *, guard, operational_compatibility):
        self.operational_compatibility = deepcopy(operational_compatibility)
        super().__init__(marker_pin, guard=guard)
        review = checked_compatibility(operational_compatibility, self.owner['run_identity'], guard)
        require(review['operational_sources']['indexed_storage'] == record(__file__)
                and self.owner['verifier_source'] == review['legacy_storage_verifier_source'],
                'Exact original storage creator and separately reviewed actual callback successor')

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
        review = checked_compatibility(self.operational_compatibility, self.owner['run_identity'], self.guard)
        actual = review['operational_sources']['controller']
        bound(actual)
        require(callable(verify_current_owner) and str(Path(inspect.getsourcefile(verify_current_owner)).resolve())==actual['path']
            and verify_current_owner({'proof':proof_pin,'operation':operation,'ownership':self.marker_pin,'state_revision':state['revision'],
                                     'latest':state['latest'],'winners':state['winners']}) is True,
            'Current owner/lease/nonce/mapping-release/restore-exclusion witness must verify at actual operation')
        if operation in ('complete_publication','publish_latest_after_fsynced_archive'):
            require(proof['publication_cadence_evidence']==self._expected_cadence
                    and digest(proof['publication_cadence_evidence'])==digest(self._expected_cadence),
                    'Current native owner must verify exact normal/tail/emergency packing evidence')
        # Optional incremental credit is available ONLY under this additional
        # source-reviewed current-owner witness. Without it, retain the frozen
        # full-prefix byte verification (no metadata boolean shortcut).
        self._journal_authority=None
        witness=proof.get('incremental_authoritative_journal')
        if witness is not None:
            epoch=witness['writer_epoch']
            require(witness['kind']=='ASR18_current_sole_append_journal_writer_identity_v1'
                    and witness['writer_source']==self.owner['journal_writer_source']
                    and witness['sole_current_writer'] is True
                    and witness['append_operation']=='open_ab_write_flush_fsync_only'
                    and witness['prior_bytes_never_rewritten_or_truncated'] is True
                    and epoch['kind']=='ASR18_verified_continuous_append_writer_epoch_v1'
                    and epoch['writer_source']==self.owner['journal_writer_source']
                    and type(epoch['producer_PID']) is int and epoch['producer_PID']>0
                    and type(epoch['producer_create_time']) is float and epoch['producer_create_time']>0
                    and type(epoch['session_owner_nonce']) is str and len(epoch['session_owner_nonce'])>=32,
                    'Exact sole append-only implementation/owner; idle or saved metadata alone cannot grant credit')
            bound(witness['writer_source']);bound(epoch['configuration']);safe(witness['path'])
            self._journal_authority=deepcopy(witness)
        else:self._journal_cache=None
        return proof

