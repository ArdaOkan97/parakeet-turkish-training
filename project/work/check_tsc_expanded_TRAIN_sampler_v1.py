"""Tiny real disk carrier/native cycler checks; synthetic clock policy only."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'outputs/parakeet_turkish'
sys.path[:0] = [str(HERE), str(ROOT / 'work/research')]
from check_TSC_uniform_majority_sampler_v1 import (
    row, CountRows, cycler, prepare_indexed_rows, TSCDomainBalancedBucketSampler,
    TSCUniformFullSampler, TSCUniformMajoritySampler, canonical,
)
from tsc_uniform_majority_ptfocus_sampler_v1 import TSCUniformMajorityPTFocusSampler
import tsc_uniform_encoder_schedule_sampler_v1 as old
import tsc_expanded_TRAIN_sampler_v1 as new
from expanded_training_rows_v1 import prepare_indexed_rows as prepare_expanded


def main():
    checks, rejections = [], []
    def ok(label, value):
        assert value, label
        checks.append(label)
    def bad(label, call):
        try: call()
        except (ValueError, KeyError): rejections.append(label); return
        raise AssertionError('Unexpected admission: ' + label)
    tr = [row('F' + str(i), group='fleurs') for i in range(36)]
    tr += [row('C' + str(i)) for i in range(24)] + [row('T' + str(i), group='tsc') for i in range(60)]
    replay = [row(lang + str(i), lang, 'fleurs') for lang in ('en', 'es', 'it', 'fr', 'de') for i in range(4)]
    pt = [row('pt' + str(i), 'pt', 'fleurs') for i in range(4)]
    kwargs = dict(seed=11, replay_seed=22, batch_size=8, accumulation=2, bucket_size=8,
                  replay_fraction=.5, fleurs_fraction=.3, cycler_factory=cycler, cycler_source_sha256='2' * 64)
    domain = TSCDomainBalancedBucketSampler(tr, replay, sampling_mode='domain30', **kwargs)
    for _ in range(5): domain.next_batches()
    original_old_constants = (old.SOURCE_END_STEP, old.TARGET_POOL_CLIPS, old.MATCHED_UPDATES)
    constant_names = ('SOURCE_STEP', 'UPDATES', 'END_STEP', 'BASE_POOL', 'EXPANDED_POOL',
                      'UNIFORM_ORIGIN', 'QUOTA_ORIGIN', 'PT_ORIGIN',
                      'MATCHED_ORIGIN', 'MATCHED_UPDATES', 'MATCHED_END')
    original_new_constants = {name: getattr(new, name) for name in constant_names}
    with tempfile.TemporaryDirectory(prefix='expanded_TRAIN_sampler_tiny_', dir=ROOT / 'work') as folder:
        root = Path(folder)
        base_rows = prepare_indexed_rows(iter(tr), root / 'original', expected_count=120,
            provenance={'synthetic_fixture_only': True, 'training_admission_complete': True,
                        'all_rows_verified': True, 'verified_clips': 120})
        catalog = CountRows(base_rows)
        expanded_rows = None
        try:
            params = dict(portuguese_replay=pt, portuguese_seed=33, sampling_mode='uniform', **kwargs)
            uniform = TSCUniformFullSampler(catalog, replay, **params)
            uniform.migrate_source_state(domain.state_dict(), domain.identity)
            for _ in range(3): uniform.next_batches()
            params['replay_fraction'] = .125
            majority = TSCUniformMajoritySampler(catalog, replay, **params)
            majority.migrate_closed_state(uniform.state_dict(), uniform.identity)
            for _ in range(2): majority.next_batches()
            focus = TSCUniformMajorityPTFocusSampler(catalog, replay, **params)
            focus.migrate_closed_state(majority.state_dict(), majority.identity)
            for _ in range(5): focus.next_batches()
            focused = focus.state_dict()
            # The actual production constants reject the toy15 source.
            parent = old.TSCUniformEncoderScheduleSampler(catalog, replay, **params)
            parent.source_prefix = deepcopy(focused['source_prefix'])
            parent.quota_transition = deepcopy(focused['quota_transition'])
            parent.PT_focus_transition = deepcopy(focused['PT_focus_transition'])
            parent.parent_majority_identity = deepcopy(focus.parent_majority_identity)
            parent.identity = deepcopy(focused['identity'])
            bad('real frozen source25109 policy rejects toy15', lambda: parent.load_state_dict(focused))
            old.SOURCE_END_STEP, old.TARGET_POOL_CLIPS, old.MATCHED_UPDATES = 15, 120, 9
            parent.load_state_dict(focused)
            parent.begin_matched_cycle({'source_step': 15, 'encoder_schedule_updates': 9,
                'configured_end_step': 24, 'parent_identity': {'domain_sampling_identity': focused['identity']},
                'source_checkpoint': {'path': 'synthetic_original', 'sha256': 'a' * 64, 'bytes': 1}})
            for _ in range(6): parent.next_batches()
            source = parent.state_dict()
            parent_identity = deepcopy(parent.identity)
            source_sha = canonical(source)
            bad('real expanded constructor requires original253798', lambda:
                new.TSCExpandedTrainSampler(catalog, catalog, replay, data_arm='base', **params))
            synthetic_constants = dict(SOURCE_STEP=21, UPDATES=11, END_STEP=32, BASE_POOL=120,
                EXPANDED_POOL=150, UNIFORM_ORIGIN=5, QUOTA_ORIGIN=8, PT_ORIGIN=10,
                MATCHED_ORIGIN=15, MATCHED_UPDATES=9, MATCHED_END=24)
            for name, value in synthetic_constants.items(): setattr(new, name, value)
            ws = [{**row('W' + str(i)), 'dataset': 'disco-eth/WorldSpeech',
                   'dataset_revision': 'f389569345a04eb8ac8fa9405b0d5048ad74b8b3',
                   'dataset_config': 'tr_tr'} for i in range(30)]
            expanded_rows = prepare_expanded(iter([*tr, *ws]), root / 'expanded', expected_count=150,
                provenance={'synthetic_fixture_only': True, 'training_admission_complete': True,
                    'all_rows_verified': True, 'verified_clips': 150,
                    'original_carrier_fingerprint': catalog.fingerprint,
                    'original_rows_preserved_value_for_value_in_same_prefix_order': True})
            expanded = CountRows(expanded_rows)
            def context(arm):
                return {'source_step': 21, 'data_arm': arm, 'data_expansion_updates': 11,
                        'configured_end_step': 32, 'TR_pool_clips': 120 if arm == 'base' else 150,
                        'parent_identity': {'domain_sampling_identity': parent_identity},
                        'source_checkpoint': {'path': 'synthetic_BEST21_opaque', 'sha256': 'b' * 64, 'bytes': 1}}
            def fresh(arm):
                return new.TSCExpandedTrainSampler(catalog, catalog if arm == 'base' else expanded,
                    replay, data_arm=arm, **params)
            values = {}
            for arm in ('base', 'expanded'):
                value = fresh(arm)
                before = (catalog.reads, expanded.reads)
                result = value.migrate_closed_state(source, parent_identity, context(arm))
                ok(arm + ': migration makes zero TR row decodes', before == (catalog.reads, expanded.reads))
                ok(arm + ': source input remains byte-value unchanged', canonical(source) == source_sha)
                saved = value.state_dict()
                ok(arm + ': all six replay queue states and RNGs exact at boundary',
                   saved['replay'] == source['replay'] and saved['replay_rng'] == source['replay_rng']
                   and saved['Portuguese_rng'] == source['Portuguese_rng'])
                ok(arm + ': TR RNG copied but new queue not shuffled/drawn',
                   saved['turkish_rng'] == source['turkish_rng']
                   and saved['turkish_streams']['uniform']['draws'] == 0
                   and saved['turkish_streams']['uniform']['cycles_started'] == 0)
                coverage = value.coverage()['data_expansion_epoch']
                ok(arm + ': fresh local coverage and inherited eval epoch offset', coverage['unique'] == 0
                   and coverage['local_updates'] == 0 and value.local_turkish_epoch == 0
                   and value.turkish_epoch == parent.turkish_epoch and value._parent is None)
                ok(arm + ': genuine cumulative TR counters retained separately',
                   value.tr_draws == parent.tr_draws and value.tr_unique == parent.tr_unique == 120)
                ok(arm + ': compact source prefix parks digest rather than old row-ID lists',
                   'source_uniform_stream_sha256' in value.source_prefix
                   and 'source_group_seen' not in value.source_prefix and 'turkish_streams' not in value.source_prefix)
                values[arm] = value
            traces = {'base': [], 'expanded': []}
            def ids(batches): return [[r['source_id'] for r in batch] for batch in batches]
            for _ in range(4):
                for arm, value in values.items():
                    before = (catalog.reads if arm == 'base' else expanded.reads)
                    batches = value.next_batches()
                    after = (catalog.reads if arm == 'base' else expanded.reads)
                    ok(arm + ': exactly14 rows materialized per update', after - before == 14)
                    ok(arm + ': physical8TR then6TR+oneoldfive+onePT',
                       len(batches) == 2 and [len(x) for x in batches] == [8, 8]
                       and all(r['language'] == 'tr' for r in batches[0])
                       and [r['language'] for r in batches[1]][-1] == 'pt'
                       and batches[1][-2]['language'] in new.OLD_LANGUAGES)
                    traces[arm].append([r['source_id'] for r in batches[1][-2:]])
            ok('both data arms have identical independent replay trace', traces['base'] == traces['expanded'])
            for arm, value in values.items():
                saved = value.state_dict()
                owncontext = {**context(arm), 'own_sampling_identity': deepcopy(value.identity)}
                restored = fresh(arm)
                before = (catalog.reads, expanded.reads)
                evidence = restored.initialize_from_own_state(saved, owncontext, parent_identity)
                ok(arm + ': own exact bootstrap without sourcecheckpoint or TR decode',
                   evidence['original40750_checkpoint_reloaded'] is False
                   and before == (catalog.reads, expanded.reads) and restored.state_dict() == saved)
                for _ in range(7):
                    assert ids(value.next_batches()) == ids(restored.next_batches())
                ok(arm + ': exact future batches and final RNG/counts equal', value.state_dict() == restored.state_dict())
                final = value.coverage()['data_expansion_epoch']
                ok(arm + ': fixed matched end and truthful unique/tail', value.updates == 32
                   and final['draws'] == 154 and final['unique'] == (120 if arm == 'base' else 150)
                   and final['repeat_visits'] == (34 if arm == 'base' else 4))
                ok(arm + ': cumulative counters do not relabel old rows as fresh coverage',
                   value.tr_draws == parent.tr_draws + 154 and value.tr_unique == (120 if arm == 'base' else 150))
                bad(arm + ': no update beyond matched budget', value.next_batches)
                for field in ('updates', 'source_group_draws', 'replay_language_counts', 'source_prefix'):
                    broken = deepcopy(saved)
                    if field == 'updates': broken[field] += 1
                    elif field == 'source_prefix': broken[field]['prior_uniform_draws'] += 1
                    else: broken[field][next(iter(broken[field]))] += 1
                    bad(arm + ': reject corrupt ' + field, lambda x=broken: restored.load_state_dict(x))
                missing = dict(owncontext); missing.pop('own_sampling_identity')
                bad(arm + ': source state cannot self-authorize exact identity', lambda:
                    fresh(arm).initialize_from_own_state(saved, missing, parent_identity))
                changed = deepcopy(owncontext)
                changed['source_checkpoint']['sha256'] = 'c' * 64
                bad(arm + ': source checkpoint substitution rejected on exact', lambda:
                    fresh(arm).initialize_from_own_state(saved, changed, parent_identity))
            ok('WorldSpeech is its own truthful source group',
               'worldspeech' in values['expanded'].coverage()['groups']
               and values['expanded'].coverage()['groups']['worldspeech']['unique_clips_seen'] == 30)
            ok('production matched arithmetic', 25198 * 14 == 352772 and 352772 - 352768 == 4
               and 40750 + 25198 == 65948)
            ok('No ML/GPU/Trackio imports', not any(x in sys.modules for x in ('torch', 'nemo', 'trackio')))
        finally:
            old.SOURCE_END_STEP, old.TARGET_POOL_CLIPS, old.MATCHED_UPDATES = original_old_constants
            for name, value in original_new_constants.items(): setattr(new, name, value)
            if expanded_rows is not None: expanded_rows.close()
            base_rows.close()
    source_path = HERE / 'tsc_expanded_TRAIN_sampler_v1.py'
    report = {'schema': 1, 'kind': 'matched_expanded_TRAIN_sampler_tiny_stdlib_verification_v1',
        'passed': True, 'checks_passed': len(checks), 'checks': checks,
        'rejections_passed': len(rejections), 'rejections': rejections,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'scope': 'Actual disk-backed120+30 synthetic metadata rows and frozen native cyclers/loaders. Production25109/253798 policies reject toy inputs first; clock/pool constants then explicitly substituted only for the tiny fixture. No actual checkpoint/corpus/audio/ML/GPU admission.',
        'both_arms_identical_replay_trace': True, 'source40750_tensor_reload_on_own_exact': False,
        'original_queue_and_seen_parked_not_reinterpreted': True}
    path = HERE / 'TSC_expanded_TRAIN_sampler_tiny_stdlib_verification_v2.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'rejections': len(rejections),
        'source_sha256': report['source_sha256'], 'report': str(path),
        'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == '__main__': main()
