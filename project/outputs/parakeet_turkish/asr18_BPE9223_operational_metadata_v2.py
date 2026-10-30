"""Bounded original-or-archive metadata reader for explicitly reviewed recovery.

No missing original path is recreated. The registered indexed archive resolver
provides exact original bytes with SHA/EOF checks; tensor payloads never enter
this JSON reader. Configuration is process-local and immutable for this worker.
"""
import json
from pathlib import Path
from bpe_broad_audio_v1 import require
from fastmss_phase_aware_plan_bridge_v7 import read_bound as direct_read

_RESOLVER = None
_BINDING = None

def configure(*, storage_owner, control_root, identity, operational_compatibility,
              current_native_resource_closure, guard, metadata_phase):
    global _RESOLVER, _BINDING
    binding = (storage_owner, str(control_root), identity, operational_compatibility,
               current_native_resource_closure)
    require(_RESOLVER is None or binding == _BINDING, 'One exact operational metadata authority per process')
    if _RESOLVER is None:
        from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore
        from asr18_BPE9223_metadata_resolver_v3 import MetadataResolver
        store = IndexedOwnedFutureCheckpointStore(storage_owner, guard=guard, operational_compatibility=operational_compatibility)
        _RESOLVER = MetadataResolver(store, control_root=Path(control_root), guard=guard,
            metadata_phase=metadata_phase, identity=identity,
            operational_compatibility=operational_compatibility,
            current_native_resource_closure=current_native_resource_closure)
        _BINDING = binding
    return _RESOLVER

def read_bound(pin, guard=lambda: None):
    if _RESOLVER is None or Path(pin['path']).exists():
        return direct_read(pin, guard)
    require(type(pin) is dict and set(pin) == {'path','bytes','sha256'}
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= 16*1024**2,
            'Only exact bounded original JSON metadata, never tensor payload')
    with _RESOLVER(pin) as stream:
        raw = bytearray()
        while len(raw) < pin['bytes']:
            guard(); block = stream.read(min(1024**2, pin['bytes'] - len(raw)))
            require(block, 'Archived original JSON truncated'); raw.extend(block)
        require(stream.read(1) == b'', 'Archived original JSON complete EOF')
    def unique(pairs):
        value = {}
        for key,item in pairs:
            require(key not in value, 'Duplicate original metadata key'); value[key] = item
        return value
    def invalid(value): raise ValueError('Nonfinite original metadata: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
