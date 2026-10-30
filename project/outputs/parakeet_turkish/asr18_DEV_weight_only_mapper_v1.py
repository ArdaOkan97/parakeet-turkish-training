"""Selective readonly DEV weights/RNG transport; no Adam tensor views.

Full raw-payload hashing verifies immutable checkpoint integrity, but only the
207 named weights and Torch/CUDA RNG descriptors become CPU tensor views.
Optimizer scalar clocks and sampler metadata are read without constructing an
optimizer, a sampler, replay state, or any saved Adam tensor object.
"""
import gc
import hashlib
import json
import math
import mmap
from pathlib import Path
import sys

from bpe_broad_production_state_v1 import digest, require
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree, select

ROOT = Path(r'E:\gpu_training_scratchpad')
CHUNK = 4 * 1024**2


def record(path, guard=lambda: None):
    path = Path(path).resolve(); before = path.stat(); hashed = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(CHUNK):
            guard(); hashed.update(block)
    require((before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns),
            'Checkpoint changed during bounded integrity hash')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': hashed.hexdigest()}


def _dictionary(node):
    require(type(node) is list and len(node) == 2 and node[0] == 'dict', 'Exact tagged saved dictionary')
    result = {}
    for pair in node[1]:
        key = decode_scalar_tree(pair[0])
        require(type(key) in (str, int) and key not in result, 'Unique nonboolean scalar saved key')
        result[key] = pair[1]
    return result


def selection_plan(document):
    """Metadata-only compatibility, never full-run or actual tensor admission."""
    from asr18_full_run_serial_DEV_v2 import checkpoint_format_view
    view = checkpoint_format_view(document)
    tree = document['tree']; top = _dictionary(tree); entries = document['tensors']
    weights = _dictionary(top['weights'])
    require(len(weights) == 207, 'Exact207 selected weight names')
    selected = {}
    def tensor(node, dtype):
        require(type(node) is list and len(node) == 2 and node[0] == 'tensor'
                and type(node[1]) is int and 0 <= node[1] < len(entries), 'Actual selected tensor index')
        index = node[1]; entry = entries[index]
        require(index not in selected and entry['dtype'] == dtype, 'Distinct selected weight/RNG tensors')
        width = 4 if dtype == 'float32' else 1
        require(entry['bytes'] == math.prod(entry['shape']) * width, 'Selected dtype/shape byte geometry')
        selected[index] = entry
        return index
    names = {name: tensor(node, 'float32') for name, node in weights.items()}
    torch_rng = tensor(top['torch_rng'], 'uint8')
    cuda_rng = [tensor(node, 'uint8') for node in top['cuda_rng'][1]]
    adam_indices = set()
    def visit(node):
        kind, value = node
        if kind == 'tensor':
            require(type(value) is int and 0 <= value < len(entries), 'Actual Adam tensor descriptor index')
            adam_indices.add(value)
        elif kind in ('tuple', 'list'):
            for child in value: visit(child)
        elif kind == 'dict':
            for key, child in value: visit(key); visit(child)
        else: require(kind == 'scalar', 'Exact metadata tree kind')
    visit(select(tree, 'optimizer'))
    require(adam_indices and not adam_indices.intersection(selected), 'No optimizer tensor can enter DEV selection')
    scalar = lambda *path: decode_scalar_tree(select(tree, *path))
    value = {
        'diagnostic_only': scalar('diagnostic_only'), 'weights': names,
        'optimizer': {'updates': scalar('optimizer', 'updates'), 'clocks': scalar('optimizer', 'clocks')},
        'sampler': scalar('sampler'), 'python_rng': scalar('python_rng'), 'numpy_rng': scalar('numpy_rng'),
        'torch_rng': torch_rng, 'cuda_rng': cuda_rng,
    }
    return {'kind': 'ASR18_DEV_selected207weights_and_all4RNG_metadata_plan_v1',
            'format_view': view, 'selected_entries': selected, 'value': value,
            'selected_weight_count': 207, 'selected_tensor_count': len(selected),
            'excluded_optimizer_tensor_count': len(adam_indices),
            'selected_optimizer_tensor_indices': [], 'actual_tensors_loaded': False,
            'full_run_checkpoint_admitted': False}


class WeightOnlyCheckpoint:
    """Same .value/.close surface as generic mapper, selected views only."""
    def __init__(self, binding, *, torch, guard):
        self.value = self.mapping = self.stream = None; self.closed = False
        path = Path(binding['path']).resolve()
        require(path.is_relative_to(ROOT.resolve()) and 0 < path.stat().st_size <= 16*1024**2
                and record(path, guard) == binding and sys.byteorder == 'little', 'Exact E saved metadata')
        document = json.loads(path.read_text(encoding='utf-8'))
        plan = selection_plan(document); payload = document['payload']; target = Path(payload['path']).resolve()
        require(target == path.with_suffix('.tensors.bin') and record(target, guard) == payload,
                'Complete saved payload integrity before selective weight mapping')
        views = {}
        try:
            self.stream = target.open('rb'); self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
            for index, entry in plan['selected_entries'].items():
                guard(); offset, size = entry['offset'], entry['bytes']; hashed = hashlib.sha256()
                for low in range(offset, offset + size, CHUNK):
                    guard(); hashed.update(self.mapping[low:min(low+CHUNK, offset+size)])
                require(hashed.hexdigest() == entry['sha256'], 'Selected weight/RNG bytes differ')
                views[index] = torch.frombuffer(self.mapping, dtype=getattr(torch, entry['dtype']),
                    count=math.prod(entry['shape']), offset=offset).reshape(entry['shape'])
            value = plan['value']
            self.value = {**value, 'weights': {name: views[index] for name, index in value['weights'].items()},
                          'torch_rng': views[value['torch_rng']], 'cuda_rng': [views[i] for i in value['cuda_rng']]}
            self.selection = {'weights': 207, 'tensor_views': len(views), 'Adam_tensor_views': 0,
                              'optimizer_objects_constructed': 0, 'plan_sha256': digest(plan),
                              'maximum_tensor_hash_slice_bytes': CHUNK}
            views.clear()
            require(record(path, guard) == binding, 'Metadata changed during selected mapping')
        except BaseException:
            self.value = None; views.clear(); gc.collect()
            if self.mapping is not None: self.mapping.close(); self.mapping = None
            if self.stream is not None: self.stream.close(); self.stream = None
            raise

    def close(self):
        if self.closed: return
        self.value = None; gc.collect()
        # The caller must drop every borrowed selected weight/RNG view first.
        if self.mapping is not None: self.mapping.close(); self.mapping = None
        if self.stream is not None: self.stream.close(); self.stream = None
        self.closed = True
