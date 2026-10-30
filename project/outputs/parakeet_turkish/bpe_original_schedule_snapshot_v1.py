"""Read the tensor-free original BPE64250 schedule inputs, never a new frontier."""
from copy import deepcopy
import json, math, random
from pathlib import Path
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest

KIND = 'ASR18_original_BPE64250_schedule_snapshot_v1'
RNG_KEYS = ('turkish_rng', 'replay_rng', 'Portuguese_rng')


def python_random_state(value):
    """Restore only the explicit JSON form of Python Random's version3 state."""
    require(type(value) in (list, tuple) and len(value) == 3 and type(value[0]) is int and value[0] == 3,
            'Explicit Python Random version3 state required')
    integers = value[1]
    require(type(integers) in (list, tuple) and len(integers) == 625
            and all(type(v) is int and 0 <= v < 2**32 for v in integers[:-1])
            and type(integers[-1]) is int and 0 <= integers[-1] <= 624,
            'Exact625-integer Python Random state required')
    gaussian = value[2]
    require(gaussian is None or type(gaussian) is float and math.isfinite(gaussian),
            'Finite original Gaussian cache required')
    state = (3, tuple(integers), gaussian)
    verifier = random.Random(); verifier.setstate(state)
    require(verifier.getstate() == state and digest(state) == digest(value), 'Lossless original RNG representation required')
    return state


def load_original_snapshot(binding, *, expected_sources, guard=lambda: None):
    guard()
    require(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}
            and 0 < binding['bytes'] <= 2*1024**2 and record(binding['path']) == binding,
            'Exact bounded original schedule snapshot required')
    value = json.loads(Path(binding['path']).read_text(encoding='utf-8'))
    require(set(value) == {'kind', 'original_clocks', 'source_records', 'snapshot', 'snapshot_sha256',
                           'authoritative_draws_advanced', 'optimizer_or_RNG_reset', 'extraction_source'}
            and value['kind'] == KIND and value['original_clocks'] == [64250, 2000, 62250]
            and value['source_records'] == expected_sources and value['authoritative_draws_advanced'] is False
            and value['optimizer_or_RNG_reset'] is False, 'Exact original lineage, scope and clocks required')
    snapshot = deepcopy(value['snapshot'])
    require(set(snapshot) == {'source_step', 'source_identity', 'replay', 'replay_language_counts', *RNG_KEYS}
            and snapshot['source_step'] == 64250 and snapshot['source_identity'] == expected_sources['bpe_parent']
            and digest(snapshot) == value['snapshot_sha256'], 'Whole original carried schedule identity required')
    require(set(snapshot['replay']) == set(snapshot['replay_language_counts']) == {'en', 'es', 'it', 'fr', 'de', 'pt'}
            and all(type(n) is int and n >= 0 for n in snapshot['replay_language_counts'].values()),
            'Original six replay queues and counters required')
    for key in RNG_KEYS: snapshot[key] = python_random_state(snapshot[key])
    guard()
    return snapshot
