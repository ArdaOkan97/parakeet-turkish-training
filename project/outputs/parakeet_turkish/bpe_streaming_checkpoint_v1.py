"""Bounded named-tensor checkpoint transport; no full CPU weight snapshot.

JSON commits a separately fsynced, hash-bound raw tensor payload. GPU tensors
are copied through at most one 4 MiB CPU slice at a time. The reader exposes
read-only mapped CPU tensors; callers copy values into active parameters/Adam.
No module instances or executable pickle are serialized.
"""
from __future__ import annotations
import gc, hashlib, json, math, mmap, os, sys
from pathlib import Path

ROOT = Path(r'E:\gpu_training_scratchpad')
KIND = 'ASR18_streaming_named_tensor_checkpoint_v1'
CHUNK = 4 * 1024**2
DTYPES = ('float32', 'bfloat16', 'float16', 'float64', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def scoped(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Checkpoint payload and metadata must stay on designated E storage')
    return path

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha}

def write_checkpoint(path, value, *, torch, guard):
    require(sys.byteorder == 'little', 'Pinned little-endian tensor encoding required')
    metadata = scoped(path)
    payload = metadata.with_suffix('.tensors.bin')
    temporary = payload.with_suffix('.writing.bin')
    require(not any(p.exists() for p in (metadata, payload, temporary)), 'Fresh immutable checkpoint namespace required')
    metadata.parent.mkdir(parents=True, exist_ok=True)
    tensors = []

    def encode(item):
        if isinstance(item, torch.Tensor):
            require(item.layout == torch.strided and item.is_contiguous() and str(item.dtype).removeprefix('torch.') in DTYPES,
                    'Only declared contiguous strided tensors are admitted')
            index = len(tensors)
            tensors.append(item.detach())
            return ['tensor', index]
        if type(item) is dict:
            require(all(type(k) in (str, int) for k in item), 'Only exact string/integer dictionary keys are admitted')
            return ['dict', [[encode(k), encode(v)] for k, v in item.items()]]
        if type(item) in (list, tuple):
            return ['tuple' if type(item) is tuple else 'list', [encode(v) for v in item]]
        require(item is None or type(item) in (str, int, float, bool), 'Unsupported checkpoint object; no executable serialization')
        return ['scalar', item]

    tree = encode(value)
    require(0 < len(tensors) <= 4096, 'Positive bounded named-tensor population required')
    entries, full_sha, offset = [], hashlib.sha256(), 0
    try:
        with temporary.open('xb') as stream:
            for index, tensor in enumerate(tensors):
                guard()
                dtype = str(tensor.dtype).removeprefix('torch.')
                size = tensor.numel() * tensor.element_size()
                padding = (-offset) % 64
                if padding:
                    raw = bytes(padding)
                    stream.write(raw); full_sha.update(raw); offset += padding
                start, sha = offset, hashlib.sha256()
                flattened = tensor.reshape(-1).view(torch.uint8)
                for low in range(0, size, CHUNK):
                    guard()
                    # .cpu() transfers only this bounded view, never the parent.
                    cpu = flattened[low:low+CHUNK].cpu()
                    raw = cpu.numpy().tobytes(order='C')
                    require(len(raw) == min(CHUNK, size-low), 'Tensor slice byte geometry changed')
                    stream.write(raw); sha.update(raw); full_sha.update(raw); offset += len(raw)
                    del raw, cpu
                entries.append({'index': index, 'dtype': dtype, 'shape': list(tensor.shape),
                                'offset': start, 'bytes': size, 'sha256': sha.hexdigest()})
                del flattened
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(payload)
        require(payload.stat().st_size == offset, 'Flushed tensor payload length differs')
        document = {'kind': KIND, 'schema_version': 1, 'endianness': 'little',
                    'source': record(__file__), 'maximum_CPU_tensor_slice_bytes': CHUNK,
                    'payload': {'path': str(payload), 'bytes': offset, 'sha256': full_sha.hexdigest()},
                    'tensors': entries, 'tree': tree, 'dynamic_module_serialization': False}
        raw = canonical(document)
        require(len(raw) <= 16 * 1024**2, 'Checkpoint metadata bound exceeded')
        # Exclusive metadata creation is the commit marker, after payload fsync.
        with metadata.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        del tensors, tree, document, raw
        gc.collect()
        return record(metadata)
    except BaseException:
        # Preserve incomplete files as evidence. No previous checkpoint is edited.
        raise

class MappedCheckpoint:
    def __init__(self, binding, *, torch, guard):
        self.value = None
        self.mapping = None
        self.stream = None
        self.closed = False
        path = scoped(binding['path'])
        require(path.stat().st_size <= 16 * 1024**2 and record(path) == binding, 'Exact bounded checkpoint metadata binding required')
        document = json.loads(path.read_text(encoding='utf-8'))
        require(document['kind'] == KIND and document['schema_version'] == 1 and document['endianness'] == sys.byteorder == 'little'
                and document['source'] == record(__file__) and document['maximum_CPU_tensor_slice_bytes'] == CHUNK
                and document['dynamic_module_serialization'] is False, 'Pinned non-executable checkpoint schema/source differs')
        payload = document['payload']; target = scoped(payload['path'])
        require(target.parent == path.parent and target == path.with_suffix('.tensors.bin') and record(target) == payload,
                'Exact immutable tensor payload binding required')
        guard()
        self.stream = target.open('rb')
        self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        tensors, previous_end = [], 0
        for index, entry in enumerate(document['tensors']):
            guard()
            shape = entry['shape']; offset = entry['offset']; size = entry['bytes']
            require(entry['index'] == index and entry['dtype'] in DTYPES and type(shape) is list
                    and all(type(v) is int and v >= 0 for v in shape)
                    and type(offset) is int and offset % 64 == 0 and offset >= previous_end
                    and type(size) is int and size >= 0 and offset + size <= payload['bytes'], 'Invalid tensor extent or geometry')
            dtype = getattr(torch, entry['dtype'])
            require(math.prod(shape) * torch.empty((), dtype=dtype).element_size() == size, 'Tensor shape/dtype bytes differ')
            sha = hashlib.sha256()
            for low in range(offset, offset+size, CHUNK):
                guard(); sha.update(self.mapping[low:min(low+CHUNK, offset+size)])
            require(sha.hexdigest() == entry['sha256'], 'Named tensor byte hash differs')
            tensor = torch.frombuffer(self.mapping, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape) if size else torch.empty(shape, dtype=dtype)
            tensors.append(tensor); previous_end = offset + size
        require(previous_end == payload['bytes'] and 0 < len(tensors) <= 4096, 'Unused payload tail or tensor population mismatch')

        def decode(node):
            require(type(node) is list and len(node) == 2, 'Invalid checkpoint tree node')
            kind, value = node
            if kind == 'tensor':
                require(type(value) is int and 0 <= value < len(tensors), 'Invalid named tensor reference')
                return tensors[value]
            if kind == 'scalar':
                require(value is None or type(value) in (str, int, float, bool), 'Unsupported checkpoint scalar')
                return value
            if kind in ('tuple', 'list'):
                values = [decode(v) for v in value]
                return tuple(values) if kind == 'tuple' else values
            require(kind == 'dict' and type(value) is list, 'Invalid checkpoint container kind')
            pairs = [(decode(k), decode(v)) for k, v in value]
            require(all(type(k) in (str, int) for k, _ in pairs) and len({k for k, _ in pairs}) == len(pairs), 'Duplicate/invalid dictionary keys')
            return dict(pairs)

        self.value = decode(document['tree'])

    def close(self):
        if self.closed:
            return
        self.value = None
        gc.collect()
        # Caller must drop every borrowed tensor/container before this boundary.
        if self.mapping is not None:
            self.mapping.close(); self.mapping = None
        if self.stream is not None:
            self.stream.close(); self.stream = None
        self.closed = True
