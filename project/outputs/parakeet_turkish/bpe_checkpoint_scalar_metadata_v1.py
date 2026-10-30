"""Restricted legacy Torch-ZIP metadata reader; never opens tensor payloads.

Only the four globals observed in the pinned compact BPE source are accepted.
Tensor/storage objects become inert descriptors. This cannot restore weights,
Adam tensors or a model and is not a general-purpose checkpoint loader.
"""
from collections import OrderedDict
from dataclasses import dataclass
import pickle, zipfile
from pathlib import Path
from bpe_broad_audio_v1 import record, require


@dataclass(frozen=True)
class StorageDescriptor:
    dtype: str
    key: str
    elements: int


@dataclass(frozen=True)
class TensorDescriptor:
    storage: StorageDescriptor
    offset: int
    shape: tuple
    stride: tuple


class ScalarMetadataUnpickler(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.storages = 0; self.tensors = 0

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'): return OrderedDict
        if module == 'torch' and name in ('FloatStorage', 'ByteStorage'): return name
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'): return self.rebuild_tensor
        raise ValueError('Unsupported checkpoint metadata global: '+module+'.'+name)

    def persistent_load(self, value):
        require(type(value) is tuple and len(value) == 5 and value[0] == 'storage'
                and value[1] in ('FloatStorage', 'ByteStorage') and type(value[2]) is str
                and value[3] == 'cpu' and type(value[4]) is int and 0 <= value[4] <= 100_000_000,
                'Only bounded inert CPU storage descriptions are supported')
        self.storages += 1
        require(self.storages <= 4096, 'Bounded storage metadata count required')
        return StorageDescriptor(value[1], value[2], value[4])

    def rebuild_tensor(self, storage, offset, shape, stride, requires_grad, backward_hooks, metadata=None):
        require(type(storage) is StorageDescriptor and type(offset) is int and 0 <= offset <= storage.elements
                and type(shape) is tuple and type(stride) is tuple and len(shape) == len(stride) <= 8
                and all(type(n) is int and n >= 0 for n in (*shape, *stride))
                and requires_grad is False and type(backward_hooks) is OrderedDict and not backward_hooks
                and metadata is None, 'Only exact inert checkpoint tensor descriptors are supported')
        self.tensors += 1
        require(self.tensors <= 4096, 'Bounded tensor metadata count required')
        return TensorDescriptor(storage, offset, shape, stride)


def read_scalar_metadata(binding, *, expected_pickle_bytes, guard=lambda: None):
    require(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}
            and type(expected_pickle_bytes) is int and 0 < expected_pickle_bytes <= 80*1024**2,
            'Explicit measured metadata bound required')
    guard(); require(record(binding['path']) == binding, 'Exact pinned legacy checkpoint required')
    path = Path(binding['path']); before = (path.stat().st_size, path.stat().st_mtime_ns)
    with zipfile.ZipFile(path) as archive:
        names = [i for i in archive.infolist() if i.filename.endswith('/data.pkl')]
        require(len(names) == 1 and names[0].file_size == expected_pickle_bytes
                and names[0].compress_type == zipfile.ZIP_STORED, 'Exact stored metadata entry required')
        with archive.open(names[0]) as stream:
            unpickler = ScalarMetadataUnpickler(stream); value = unpickler.load()
            require(stream.read(1) == b'' and type(value) is dict, 'Single complete checkpoint dictionary required')
            counts = {'inert_tensor_descriptors': unpickler.tensors, 'inert_storage_descriptors': unpickler.storages}
    guard(); require(before == (path.stat().st_size, path.stat().st_mtime_ns), 'Checkpoint changed during scalar read')
    return value, {'source': record(__file__), 'checkpoint': binding, 'metadata_bytes': expected_pickle_bytes,
                   **counts, 'tensor_storage_payloads_opened': 0, 'torch_or_numpy_import_required': False}
