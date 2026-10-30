"""Create a new inactive publication-safe coordinator; preserve v2."""
from pathlib import Path
import json,sys
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    old=S/'bpe_production_window_coordinator_v2.py';target=S/'bpe_production_window_coordinator_v3.py';require(not target.exists(),'Fresh inactive coordinator successor required')
    text=old.read_text();begin=text.index('    def publish_checkpoint(')
    text=text[:begin]+'''    def publish_checkpoint(self,*,write_coherent_checkpoint,pointer_path,checkpoint_identity,diagnostic_only):
        serial();require(not self.poisoned and not self.optimizer.poisoned and not self.schedule.poisoned and self.index>0
            and type(diagnostic_only) is bool,'Explicit scope and coherent committed state required before publishing')
        state=deepcopy(self.schedule.state_dict());updates=self.optimizer.updates;journal=deepcopy(self.journal.binding())
        caller_identity=deepcopy(checkpoint_identity);index=self.index;plan_digest=digest(self.plan)
        last=deepcopy(self.plan['updates'][index-1])
        require(updates==self.schedule.committed_updates() and digest(state)==last['after_sampler_sha256'],
                'Checkpoint state/frontier differs from last actual commit')
        identity={'kind':'ASR18_window_checkpoint_live_identity_v1','caller_identity':caller_identity,
            'window':{'plan_sha256':plan_digest,'committed_window_updates':index,'last_transaction_sha256':last['native_transaction_sha256'],
                'sampler_state_sha256':digest(state),'optimizer_updates':updates,'committed_journal':journal},'diagnostic_only':diagnostic_only}
        # Never give a writer a mutable reference to the comparison snapshots.
        saved=write_coherent_checkpoint(deepcopy(state),deepcopy(identity))
        binding(saved);require(saved['bytes']<=16*1024**2,'Bounded returned checkpoint metadata required')
        metadata=json.loads(Path(saved['path']).read_text())
        from bpe_streaming_checkpoint_semantics_v1 import verify_saved_metadata
        proof=verify_saved_metadata(metadata,identity=identity,optimizer_updates=updates,sampler_state=state,
            journal_prefix=journal,diagnostic_only=diagnostic_only)
        binding(metadata['payload'])
        require(self.optimizer.updates==updates and self.schedule.state_dict()==state and self.journal.binding()==journal
            and self.index==index and digest(self.plan)==plan_digest and checkpoint_identity==caller_identity,
            'Checkpoint writer changed scientific state, identity, window position or journal')
        pointer=Path(pointer_path).resolve();require(pointer.is_relative_to(ROOT.resolve()),'Latest pointer E-only')
        value={'kind':'ASR18_production_window_completed_checkpoint_pointer_v2','checkpoint':saved,'payload':metadata['payload'],
            'committed_updates':updates,'sampler_state_sha256':digest(state),'journal_prefix':journal,'identity':identity,
            'writer_returned_and_payload_verified':True,'saved_semantic_state_verified':proof,'committed_window_updates':index,
            'all_window_updates_committed':index==len(self.plan['updates']),'diagnostic_only':diagnostic_only}
        write_json(pointer,value);return value
'''
    target.write_text(text)
    write_json(A/'production_window_coordinator_publication_successor_v3.json',{'source':record(target),'prior_preserved_source':record(old),
        'semantic_verifier':record(S/'bpe_streaming_checkpoint_semantics_v1.py'),'source_bug_control':record(A.parent/'FastMSS_integration_v1/production_window_stale_return_control_01/report.json'),
        'native_consume_or_sampler_math_changed':False,'inactive_source_only':True,'production_enabled':False,'ROOT_review_pending':True})
    print(json.dumps({'inactive_coordinator':record(target),'production_enabled':False}))
if __name__=='__main__':main()
