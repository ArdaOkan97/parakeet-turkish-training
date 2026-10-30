"""Read-only saved-state persistence check against independent PRE-save hashes.

No tensor, model, optimizer or numerical package is imported. All payload bytes
are independently hashed in <=4MiB reads; the named tensor tree is reconstructed
as shape/dtype/byte/hash descriptors. This does not restore a GPU checkpoint.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from bpe_streaming_checkpoint_semantics_v2 import verify_saved_metadata

CHUNK = 4 * 1024**2
WIDTHS = {'float32':4, 'bfloat16':2, 'float16':2, 'float64':8, 'int64':8,
          'int32':4, 'int16':2, 'int8':1, 'uint8':1, 'bool':1}
ROOT_FIELDS = {'kind', 'diagnostic_only', 'identity', 'source_provenance', 'weights',
               'optimizer', 'sampler', 'augmentation', 'python_rng', 'torch_rng',
               'cuda_rng', 'numpy_rng'}


def normalize_fingerprint(value):
    """Restore the one integer-keyed fingerprint map after JSON transport."""
    result = deepcopy(value)
    require(set(result) == {'state_sha256', 'state'}
            and result['state']['kind'] == 'ASR18_actual_live207_threeAdam_all4RNG_fingerprint_v1',
            'Exact independent live fingerprint contract required')
    for partition in result['state']['Adam_partitions']:
        states = {}
        for key, state in partition['state'].items():
            require(type(key) is int and key >= 0 or type(key) is str and re.fullmatch(r'0|[1-9][0-9]*',key),
                    'Canonical nonboolean Adam state index required')
            index = int(key)
            require(index not in states, 'Duplicate Adam index after JSON normalization')
            states[index] = state
        partition['state'] = states
    require(result['state_sha256'] == digest(result['state']),
            'Independent PRE-save fingerprint changed in transport')
    return result


def _saved_view(binding, *, guard):
    path = Path(binding['path']).resolve()
    require(path.is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve())
            and 0 < path.stat().st_size <= 16*1024**2 and record(path) == binding,
            'Exact bounded saved metadata required')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    require(metadata['kind'] == 'ASR18_streaming_named_tensor_checkpoint_v1'
            and type(metadata['schema_version']) is int and metadata['schema_version'] == 1
            and metadata['endianness'] == 'little' and metadata['dynamic_module_serialization'] is False
            and metadata['maximum_CPU_tensor_slice_bytes'] == CHUNK
            and metadata['source'] == record(Path(__file__).with_name('bpe_streaming_checkpoint_v1.py')),
            'Exact frozen non-executable streamed writer required')
    payload = metadata['payload']; target = Path(payload['path']).resolve()
    require(target == path.with_suffix('.tensors.bin') and target.stat().st_size == payload['bytes']
            and type(payload['bytes']) is int and payload['bytes'] > 0
            and re.fullmatch(r'[0-9a-f]{64}',payload['sha256']), 'Exact committed payload pair required')
    signature = (target.stat().st_size, target.stat().st_mtime_ns)
    entries = metadata['tensors']
    require(type(entries) is list and 0 < len(entries) <= 4096, 'Bounded tensor population')
    full = hashlib.sha256(); descriptors = []; end = 0
    with target.open('rb') as stream:
        for index, entry in enumerate(entries):
            guard(); padding = (-end) % 64
            require(type(entry['index']) is int and entry['index'] == index
                    and type(entry['dtype']) is str and entry['dtype'] in WIDTHS
                    and type(entry['shape']) is list and all(type(v) is int and v >= 0 for v in entry['shape'])
                    and type(entry['offset']) is int and entry['offset'] == end+padding
                    and type(entry['bytes']) is int
                    and entry['bytes'] == math.prod(entry['shape'])*WIDTHS[entry['dtype']]
                    and entry['offset']+entry['bytes'] <= payload['bytes'],
                    'Exact consecutive/aligned saved tensor geometry required')
            raw_padding = stream.read(padding)
            require(raw_padding == bytes(padding), 'Frozen writer padding differs')
            full.update(raw_padding); end += padding
            hashed = hashlib.sha256(); remaining = entry['bytes']
            while remaining:
                guard(); block = stream.read(min(CHUNK, remaining))
                require(0 < len(block) <= CHUNK, 'Truncated saved tensor payload')
                hashed.update(block); full.update(block); remaining -= len(block); end += len(block)
                del block
            require(hashed.hexdigest() == entry['sha256'], 'Actual saved tensor bytes differ from named digest')
            descriptors.append({'shape': entry['shape'], 'dtype': 'torch.'+entry['dtype'],
                                'bytes': entry['bytes'], 'sha256': entry['sha256']})
        require(end == payload['bytes'] and not stream.read(1), 'Saved payload EOF differs')
    require(full.hexdigest() == payload['sha256']
            and (target.stat().st_size,target.stat().st_mtime_ns) == signature
            and record(path) == binding, 'Full actual payload or metadata changed during read-only verification')
    used = set(); visited = [0]
    def decode(node, depth=0):
        visited[0] += 1
        require(depth <= 64 and visited[0] <= 2_000_000 and type(node) is list and len(node) == 2,
                'Bounded exact tagged saved tree required')
        kind, value = node
        if kind == 'tensor':
            require(type(value) is int and 0 <= value < len(descriptors), 'Exact tensor reference')
            used.add(value); return deepcopy(descriptors[value])
        if kind == 'scalar':
            require(value is None or type(value) in (str,int,float,bool), 'Exact supported saved scalar')
            require(type(value) is not float or math.isfinite(value), 'Finite saved scalar')
            return value
        if kind in ('list','tuple'):
            require(type(value) is list, 'Exact saved sequence')
            result = [decode(v,depth+1) for v in value]
            return tuple(result) if kind == 'tuple' else result
        require(kind == 'dict' and type(value) is list, 'Exact saved dictionary')
        result = {}
        for pair in value:
            require(type(pair) is list and len(pair) == 2, 'Exact dictionary pair')
            key = decode(pair[0],depth+1)
            require(type(key) in (str,int) and key not in result, 'Unique nonboolean saved key')
            result[key] = decode(pair[1],depth+1)
        return result
    saved = decode(metadata['tree'])
    require(type(saved) is dict and set(saved) == ROOT_FIELDS and len(used) == len(entries),
            'Complete frozen state schema and all tensor references required')
    opt = saved['optimizer']; weights = saved['weights']
    require(type(weights) is dict and len(weights) == 207 and len(opt['optimizers']) == 3
            and type(opt['updates']) is int and opt['updates'] > 0
            and opt['clocks'] == [n+opt['updates'] for n in (64250,2000,62250)],
            'Exact named207/three mature Adam clocks required')
    partitions = []; total = 0
    for part in opt['optimizers']:
        indices = [i for group in part['param_groups'] for i in group['params']]
        require(all(type(i) is int and i >= 0 for i in indices) and len(indices) == len(set(indices))
                and set(indices) == set(part['state']), 'Exact complete indexed Adam partition')
        require(all(set(part['state'][i]) == {'step','exp_avg','exp_avg_sq'} for i in indices),
                'All three Adam state tensors required')
        partitions.append({'param_groups':part['param_groups'],'state':part['state']}); total += len(indices)
    require(total >= 207, 'Complete sliced Adam state coverage')
    state = {'kind':'ASR18_actual_live207_threeAdam_all4RNG_fingerprint_v1',
        'updates':opt['updates'],'identity':opt['identity'],'clocks':opt['clocks'],
        'weights':{name:weights[name] for name in sorted(weights)},'Adam_partitions':partitions,
        'RNG':{'python':saved['python_rng'],'numpy':saved['numpy_rng'],
               'torch':saved['torch_rng'],'CUDA':saved['cuda_rng']},
        'maximum_transfer_slice_bytes':CHUNK}
    return metadata, saved, {'state_sha256':digest(state),'state':state}, len(entries), total


def saved_fingerprint(binding, *, guard):
    """Transport-only view; no independent PRE-save equality is asserted here."""
    _, _, fingerprint, _, _ = _saved_view(binding, guard=guard)
    return fingerprint


def verify_persistence(binding, *, expected_pre_save, identity, sampler_state,
                       journal_prefix, diagnostic_only, guard):
    expected = normalize_fingerprint(expected_pre_save)
    metadata, saved, actual, count, Adam_count = _saved_view(binding, guard=guard)
    semantic = verify_saved_metadata(metadata, identity=identity,
        optimizer_updates=expected['state']['updates'], sampler_state=sampler_state,
        journal_prefix=journal_prefix, diagnostic_only=diagnostic_only)
    require(actual['state_sha256'] == expected['state_sha256'],
            'Actual saved207/threeAdam/all4RNG differs from independent PRE-save live state')
    require(digest(saved['sampler']) == digest(sampler_state)
            and digest(saved['augmentation']['committed_journal']) == digest(journal_prefix),
            'Saved actual sampler and authoritative prefix differ')
    return {'kind':'ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1','passed':True,
        'checkpoint':binding,'payload':metadata['payload'],
        'independent_pre_save_state_sha256':expected['state_sha256'],
        'saved_state_sha256':actual['state_sha256'],'optimizer_updates':expected['state']['updates'],
        'weights_verified':207,'Adam_states_verified':Adam_count,'named_tensors_verified':count,
        'sampler_state_sha256':digest(sampler_state),'journal_prefix':deepcopy(journal_prefix),
        'saved_semantic_state_verified':semantic,'maximum_payload_read_bytes':CHUNK,
        'actual_GPU_restore_performed':False,'live_model_optimizer_or_RNG_mutated_here':False,
        'independent_post_check_live_fingerprint_comparison_required':True,
        'full_fresh_GPU_restore_required_before_overall_diagnostic_pass':True,
        'source':record(__file__)}
