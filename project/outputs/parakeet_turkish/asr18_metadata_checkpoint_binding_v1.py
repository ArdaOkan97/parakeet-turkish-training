"""Frozen augmentation checkpoint metadata only; no augmenter or transport."""
from fastmss_clean_feature_contract_v1 import metadata_clean_binding
from fastmss_ordered_draw_binding_v1 import FastMSSOrderedDrawBinding
from bpe_broad_production_state_v1 import canonical
import json
class MetadataCheckpointBinding:
    def __init__(self,identity,*,guard):
        self._identity=metadata_clean_binding(identity,guard).identity
    @property
    def identity(self):return json.loads(canonical(self._identity))
    checkpoint_binding=FastMSSOrderedDrawBinding.checkpoint_binding
    verify_restore=FastMSSOrderedDrawBinding.verify_restore
