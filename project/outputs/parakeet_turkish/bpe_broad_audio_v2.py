"""Production-only reader revision: finalized file/index/receipt signatures.

Decode and 16kHz canonicalization are inherited byte-for-byte from the proven
v1 reader. New production admissions cannot re-trust modified same-size data.
"""
from bpe_broad_audio_v1 import BroadAudioReader as OriginalReader
from bpe_broad_source_identity_v1 import FinalSourceIdentity

class BroadAudioReader(OriginalReader):
    def __init__(self,*,source_inventory_binding,**kwargs):
        super().__init__(**kwargs)
        self.final_source_identity=FinalSourceIdentity(source_inventory_binding,self.plan,self.jobs)

    def verify_source(self,path):
        return self.final_source_identity.verify(path)
