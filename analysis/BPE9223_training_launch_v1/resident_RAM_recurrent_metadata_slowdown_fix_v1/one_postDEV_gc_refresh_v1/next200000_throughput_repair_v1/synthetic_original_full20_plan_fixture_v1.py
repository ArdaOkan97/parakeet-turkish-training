"""METHOD_STUB scientific owners feeding the REAL original make_plan function.
No original population, SQLite, audio, optimizer or authoritative RNG is opened.
Two frozen representative domains expose metadata callback and text-byte costs.
"""
from copy import deepcopy
import hashlib
import random
from bpe_broad_production_state_v1 import digest
from bpe_production_window_coordinator_v4 import make_plan
from bpe_training_schedule_v1 import WEAK


def stub_pin(name, size=8192):
    return {'path': 'E:/METHOD_STUB/never_opened/' + name, 'bytes': size,
            'sha256': hashlib.sha256(name.encode()).hexdigest()}


class _Cursor:
    def __init__(self):
        self.cursor = 0

    def commit(self, draws):
        self.cursor += len(draws)


class _Replay:
    def __init__(self, state):
        self.state = deepcopy(state)

    def load_state_dict(self, state):
        self.state = deepcopy(state)


class _Transaction:
    def abandon(self):
        pass  # METHOD_STUB planning transaction owns no authoritative state.


class _Schedule:
    """Only the exact owner interface consumed by original make_plan, not science."""
    def __init__(self, seed):
        rng = random.Random(seed)
        self.broad = _Cursor()
        self.cleaner = _Cursor()
        self.replay = _Replay({'updates': 204000, 'selector_rng': rng.getstate(),
            'buckets': {lang: {'queue': list(range(48)), 'rng': random.Random(seed + index).getstate()}
                        for index, lang in enumerate(('en', 'es', 'it', 'fr', 'de', 'pt'))}})
        self.phase = WEAK
        self.poisoned = False
        self.active_cursor = self.broad

    def state_dict(self):
        return {'broad_cursor': {'cursor': self.broad.cursor, 'epoch': 1,
                                 'identity': stub_pin('synthetic_full_population.json')},
                'cleaner_cursor': {'cursor': self.cleaner.cursor, 'epoch': 0},
                'carried_replay': deepcopy(self.replay.state)}

    def load_state_dict(self, state):
        self.broad.cursor = state['broad_cursor']['cursor']
        self.cleaner.cursor = state['cleaner_cursor']['cursor']
        self.replay.load_state_dict(state['carried_replay'])

    def committed_updates(self):
        return self.replay.state['updates']

    def propose(self, *, optimizer_updates, resolve_TR):
        assert optimizer_updates == self.committed_updates()
        tx = _Transaction()
        tx.before = self.state_dict()
        tx.TR_draws = [{'ordinal': self.broad.cursor + index + 1, 'epoch': 1,
                        'draw_index': self.broad.cursor + index} for index in range(14)]
        tx.TR_identity = [resolve_TR(self.phase, draw['ordinal']) for draw in tx.TR_draws]
        tx.replay_rows = [{'source_id': 'METHOD_STUB_replay_' + str(optimizer_updates) + '_' + str(index),
                           'language': ('en', 'es')[index],
                           'cache_path': 'E:/METHOD_STUB/never_opened/replay_' + str(index) + '.pt',
                           'encoder_frames': 165, 'target_tokens': 66,
                           'augmentation_role': 'replay'} for index in range(2)]
        tx.after_replay = deepcopy(self.replay.state)
        tx.after_replay['updates'] += 1
        tx.identity_digest = digest({'before': tx.before, 'draws': tx.TR_draws,
                                     'identities': tx.TR_identity, 'replay': tx.replay_rows})
        return tx


def build_original_plan(seed, shape, population_pin):
    assert shape in ('metadata_heavy', 'text_heavy')
    rng = random.Random(seed)
    phrase = 'Özgün Türkçe kaynak ve tam hedef. '
    texts = phrase * (3 if shape == 'metadata_heavy' else 20)
    target_count = 100 if shape == 'metadata_heavy' else 66
    source_pins = {name: stub_pin(name + '.sqlite', 17000000 + index)
                   for index, name in enumerate(('catalog', 'labels', 'geometry'))}

    def identity(phase, ordinal):
        assert phase == WEAK and type(ordinal) is int
        return {'ordinal': ordinal, 'source_id': 'METHOD_STUB_TR_' + str(ordinal).zfill(8), 'language': 'tr'}

    def rows(phase, draws, replay):
        result = []
        for draw in draws:
            ordinal = draw['ordinal']
            source = identity(phase, ordinal)['source_id']
            raw = {'ordinal': ordinal, 'dataset': 'alania_yodas3', 'unique_id': source,
                   'source_id': 'METHOD_STUB_recording_' + str(ordinal),
                   'text': texts + str(ordinal), 'start': 0.0, 'end': 13.143,
                   'source_artifact': 'E:/METHOD_STUB/never_opened/part_' + str(ordinal // 30) + '.tar',
                   'member_name': '0000/' + str(ordinal).zfill(8) + '.opus',
                   'member_offset': ordinal * 4096, 'member_bytes': 261223,
                   'source_component': 'alania_yodas3'}
            labels = [rng.randrange(8192) for _ in range(target_count)]
            declared = {'ordinal': ordinal, 'source_component': 'alania_yodas3', 'native_sr': 48000,
                        'native_samples': 630864, 'clean16k_samples': 210288,
                        'mel_frames': 1314, 'encoder_frames': 165, 'target_tokens': target_count,
                        'status': 'METHOD_STUB_declared',
                        'source_signatures': [{'path': pin['path'], 'bytes': pin['bytes'], 'mtime_ns': 1790000000000000000}
                                              for pin in source_pins.values()]}
            result.append({**raw, 'broad_original': deepcopy(raw),
                'recording_source_id': raw['source_id'], 'source_id': source, 'language': 'tr',
                'augmentation_role': 'student_tr', 'encoder_frames': 165, 'target_tokens': target_count,
                'tr_metadata_labels': labels, 'training_input_type': 'TR_frame_target_metadata_v1',
                'declared_geometry': declared, 'runtime_full_audio_geometry_and_heldout_check_required': True,
                'declared_frontend_geometry': {'input_samples': 210288, 'valid_mel_frames': 1314,
                    'physical_mel_frames': 1315, 'encoder_frames': 165},
                'METHOD_STUB_provenance': deepcopy(source_pins)})
        for original in replay:
            cache = stub_pin(original['cache_path'].split('/')[-1], 9800000)
            result.append({**original, 'clean_target_ids': list(range(66)),
                'full_replay_label_binding': {'cache': cache,
                    'cache_signature': {'bytes': cache['bytes'], 'mtime_ns': 1790000000000000000},
                    'source': stub_pin('bpe_replay_cache_labels_v1.py', 7200),
                    'full_label_count': 66, 'teacher_feature_shape': [165, 1024],
                    'teacher_feature_storage_decoded': False, 'label_storage_bytes_decoded': 528,
                    'whole_cache_opaque_SHA256_reverified': True, 'actual_audio_or_native_update_proved': False}})
        return result

    schedule = _Schedule(seed)
    before = deepcopy(schedule.state_dict())
    plan = make_plan(schedule, clone_schedule=lambda: _Schedule(seed), resolve_identity=identity,
                     resolve_rows=rows, feature_bytes_upper_bound=lambda phase, values: 1000000,
                     finalized_population=population_pin,
                     verify_finalized_population=lambda value: value == population_pin, maximum_updates=20)
    assert schedule.state_dict() == before
    assert len(plan['updates']) == 20
    assert all(len(entry['rows']) == 16 and len(entry['TR_draws']) == 14
               and len(entry['replay_rows']) == 2 and len(entry['contexts']) == 16 for entry in plan['updates'])
    return plan

