"""Source-only reuse of proven candidate mapping-release protocol for fullowner."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
t=(S/'asr18_full_run_native_owner_v2.py').read_text(encoding='utf-8-sig')
t=t.replace('self.identity=deepcopy(config[\'full_run_identity\']);self.last_persistence=None',
 "self.identity=deepcopy(config['full_run_identity']);self.last_persistence=None\n        self.checkpoint_writer_open=False;self.checkpoint_reader_open=False;self.checkpoint_restore_open=False;self.candidate_intent=None")
needle="            state,journal,branch=restore_state(pointer['checkpoint'],torch=self.torch,model=self.model,optimizer=self.optimizer,\n                identity=pointer['identity'],augmentation=self.binding,journal_factory=self.journal_factory,\n                journal_factory_source=self.journal_factory_source,diagnostic_only=False,guard=self.guard)"
replacement="""            self.checkpoint_restore_open=True
            try:
                state,journal,branch=restore_state(pointer['checkpoint'],torch=self.torch,model=self.model,optimizer=self.optimizer,
                    identity=pointer['identity'],augmentation=self.binding,journal_factory=self.journal_factory,
                    journal_factory_source=self.journal_factory_source,diagnostic_only=False,guard=self.guard)
            finally:self.checkpoint_restore_open=False"""
assert needle in t;t=t.replace(needle,replacement)
needle="                return save_state(intent['metadata_path'],torch=self.torch,optimizer=self.optimizer,state=saved,\n                    provenance=self.provenance,identity=identity,augmentation=self.binding,journal=self.journal,\n                    diagnostic_only=False,guard=self.guard)"
replacement="""                self.checkpoint_writer_open=True
                try:
                    return save_state(intent['metadata_path'],torch=self.torch,optimizer=self.optimizer,state=saved,
                        provenance=self.provenance,identity=identity,augmentation=self.binding,journal=self.journal,
                        diagnostic_only=False,guard=self.guard)
                finally:self.checkpoint_writer_open=False"""
assert needle in t;t=t.replace(needle,replacement)
needle="            proof=verify_persistence(pointer['checkpoint'],expected_pre_save=before,identity=pointer['identity'],sampler_state=state,\n                journal_prefix=pointer['journal_prefix'],diagnostic_only=False,guard=self.guard)"
replacement="""            self.checkpoint_reader_open=True
            try:
                proof=verify_persistence(pointer['checkpoint'],expected_pre_save=before,identity=pointer['identity'],sampler_state=state,
                    journal_prefix=pointer['journal_prefix'],diagnostic_only=False,guard=self.guard)
            finally:self.checkpoint_reader_open=False"""
assert needle in t;t=t.replace(needle,replacement)
t=t.replace("        return self.capture\n","        self.candidate_intent=deepcopy(intent)\n        return self.capture\n")
needle='    def close(self):self.context.close()'
method='''    def checkpoint_storage_release_ack(self,*,nonce,window_completion):
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

    def close(self):self.context.close()'''
assert needle in t;t=t.replace(needle,method);ast.parse(t)
out=S/'asr18_full_run_native_owner_v3.py';assert not out.exists();out.write_text(t,encoding='utf-8')
r=(R/'asr18_full_run_owned_GPU_runtime_v2.py').read_text(encoding='utf-8-sig').replace('from asr18_full_run_native_owner_v2 import','from asr18_full_run_native_owner_v3 import')
needle="            ip=folder/('idle_ack_'+str(sequence)+'.json');write_json(ip,idle)\n            write_json(folder/'response.json',{'passed':True,'sequence':sequence,'nonce':command['nonce'],\n                'window_completion':complete,'idle_ack':record(ip)})"
replacement="""            ip=folder/('idle_ack_'+str(sequence)+'.json');write_json(ip,idle)
            released=engine.checkpoint_storage_release_ack(nonce=command['nonce'],window_completion=complete)
            release_path=folder/('checkpoint_release_ack_'+str(sequence)+'.json');write_json(release_path,released)
            write_json(folder/'response.json',{'passed':True,'sequence':sequence,'nonce':command['nonce'],
                'window_completion':complete,'idle_ack':record(ip),'checkpoint_storage_release_ack':record(release_path)})"""
assert needle in r;r=r.replace(needle,replacement);ast.parse(r)
out=R/'asr18_full_run_owned_GPU_runtime_v3.py';assert not out.exists();out.write_text(r,encoding='utf-8')
print('Native3/runtime3 syntax-only source successors; no target imports')
