"""Metadata-only exact recorded-noise16room identity; original commit protocol."""
from copy import deepcopy
from bpe_broad_audio_v1 import require
from fastmss_BPE9223_recorded_room_train_v2 import policy_identity
from fastmss_ordered_draw_binding_v1 import FastMSSOrderedDrawBinding


class MetadataCheckpointBinding:
    def __init__(self,identity,*,guard):
        _,actual=policy_identity(identity['policy_file'],guard)
        require(actual==identity,'Same exact9223 recorded-noise/16room metadata, no hidden pilot adoption')
        self._identity=deepcopy(identity)
    @property
    def identity(self):return deepcopy(self._identity)
    checkpoint_binding=FastMSSOrderedDrawBinding.checkpoint_binding
    verify_restore=FastMSSOrderedDrawBinding.verify_restore
