"""Tiny actual indexed/native sampler checks; no actual corpus/PT/ML/GPU."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse
import ast
import hashlib
import json
import math
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(HERE))
from lazy_training_rows_v1 import prepare_indexed_rows
from tsc_domain_sampler_v3 import TSCDomainBalancedBucketSampler, TSC_REPO, TSC_REVISION
from tsc_uniform_full_sampler_v1 import TSCUniformFullSampler, canonical
from tsc_uniform_majority_sampler_v1 import TSCUniformMajoritySampler, KIND
from turkish_domain_sampler_v1 import FLEURS_REVISION, CV_REVISION

tree = ast.parse((HERE / 'train_lora_batched_data_v11.py').read_text(encoding='utf-8'))
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('ids_sha', 'BucketCycler')]
namespace = {'hashlib': hashlib, 'math': math}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen BucketCycler AST', 'exec'), namespace)
cycler = namespace['BucketCycler']
checks, rejections = [], []


def ok(value, message):
    if not value:
        raise AssertionError(message)
    checks.append(message)


def bad(call, message):
    try:
        call()
    except (ValueError, KeyError):
        checks.append(message)
        rejections.append(message)
        return
    raise AssertionError('Unexpected acceptance: ' + message)


def row(source, language='tr', group='cv'):
    return {'source_id': source, 'language': language,
        'dataset': 'google/fleurs' if group == 'fleurs' else TSC_REPO if group == 'tsc' else 'ysdede/commonvoice_17_tr_fixed',
        'dataset_revision': FLEURS_REVISION if group == 'fleurs' else TSC_REVISION if group == 'tsc' else CV_REVISION,
        'dataset_config': 'tr_tr' if language == 'tr' else 'pt_br' if language == 'pt' else language,
        'dataset_split': 'train', 'training_role': 'train', 'text': 'tiny training text',
        'duration': 1. + len(source) / 100, 'encoder_frames': 10 + len(source), 'target_tokens': 3,
        'audio_sha256': '0' * 64, 'label_ids_sha256': '1' * 64}


class CountRows:
    def __init__(self, parent):
        self.parent, self.reads = parent, 0
    def __getattr__(self, name):
        return getattr(self.parent, name)
    def __len__(self):
        return len(self.parent)
    def __getitem__(self, index):
        self.reads += 1
        return self.parent[index]
    def __iter__(self):
        raise AssertionError('No complete decoded TR iteration allowed')


def main(args):
    assert not args.output.exists(), 'Use a fresh proof output path'
    tr = [row('F' + str(i), group='fleurs') for i in range(36)]
    tr += [row('C' + str(i)) for i in range(24)] + [row('T' + str(i), group='tsc') for i in range(60)]
    replay = [row(lang + str(i), lang, 'fleurs') for lang in ('en', 'es', 'it', 'fr', 'de') for i in range(4)]
    pt = [row('pt' + str(i), 'pt', 'fleurs') for i in range(4)]
    kwargs = dict(seed=11, replay_seed=22, batch_size=8, accumulation=2, bucket_size=8,
        replay_fraction=.5, fleurs_fraction=.3, cycler_factory=cycler, cycler_source_sha256='2' * 64)
    domain = TSCDomainBalancedBucketSampler(tr, replay, sampling_mode='domain30', **kwargs)
    for _ in range(5):
        domain.next_batches()
    original = domain.state_dict()
    with tempfile.TemporaryDirectory(prefix='majority_index_tiny_', dir=ROOT / 'work/research') as directory:
        indexed = prepare_indexed_rows(iter(tr), Path(directory) / 'rows', expected_count=120,
            provenance={'synthetic_fixture_only': True, 'training_admission_complete': True,
                        'all_rows_verified': True, 'verified_clips': 120})
        catalog = CountRows(indexed)
        try:
            def old():
                value = TSCUniformFullSampler(catalog, replay, portuguese_replay=pt,
                    portuguese_seed=33, sampling_mode='uniform', **kwargs)
                value.migrate_source_state(original, domain.identity)
                return value
            prior = old()
            for _ in range(3):
                prior.next_batches()
            closed = prior.state_dict()
            closed_sha = canonical(closed)
            def fresh():
                return TSCUniformMajoritySampler(catalog, replay, portuguese_replay=pt,
                    portuguese_seed=33, sampling_mode='uniform', **{**kwargs, 'replay_fraction': .125})
            new = fresh()
            bad(new.next_batches, 'No unsourced majority draws')
            reads = catalog.reads
            evidence = new.migrate_closed_state(closed, prior.identity)
            initial = new.state_dict()
            ok(catalog.reads == reads, 'Construction/migration/snapshot decode zero TR dictionaries')
            ok(canonical(closed) == closed_sha, 'Original source state remains byte-equivalent canonical values')
            ok(all(initial[k] == closed[k] for k in closed if k not in ('kind', 'identity')),
               'Every source native queue/RNG/counter/prefix field preserved at boundary')
            ok(new.updates == 8 and new.transition_source_step == 8
               and new.remaining_update_budget == 7 and new.configured_end_step == 15,
               'Remaining horizon is ceil((120 minus24)/14), never a fresh pass')
            ok(new.coverage()['uniform_pass']['draws'] == 24 and new.coverage()['uniform_pass']['source_step'] == 5,
               'Original uniform coverage clock and current frontier retained')
            ok(evidence['physical_microbatch_composition'] == [[8, 0], [6, 2]], 'B8x2 packing declared accurately')
            reference = old()
            reference.load_state_dict(closed)
            before = catalog.reads
            batches = new.next_batches()
            ok(catalog.reads - before == 14, 'Only14 next TR dictionaries materialized')
            actual = [r['source_id'] for batch in batches for r in batch if r['language'] == 'tr']
            expected = [reference.turkish_streams['uniform'].next()['source_id'] for _ in range(14)]
            ok(actual == expected, 'TR next visits equal existing queue suffix without shuffle or drop')
            expected_replay = []
            for _ in range(2):
                lang = reference.replay_rng.choice(list(reference.replay))
                expected_replay.append(reference.replay[lang].next()['source_id'])
            ok([r['source_id'] for b in batches for r in b if r['language'] != 'tr'] == expected_replay,
               'Copied selector/old5/PT queues supply exactly the next two replay visits')
            ok([Counter(r['language'] == 'tr' for r in batch) for batch in batches]
               == [Counter({True: 8}), Counter({True: 6, False: 2})], 'Actual physical grouping8TR then6TR2RP')
            saved = new.state_dict()
            restored = fresh()
            restored.migrate_closed_state(closed, prior.identity)
            reads = catalog.reads
            restored.load_state_dict(saved)
            ok(restored.state_dict() == saved and catalog.reads == reads, 'Own exact snapshot restores all fields with zero TR decodes')
            for _ in range(6):
                ok([[r['source_id'] for r in b] for b in new.next_batches()]
                   == [[r['source_id'] for r in b] for b in restored.next_batches()],
                   'Future exact TR and six replay sequences equal')
            final = new.state_dict()
            cover = new.coverage()['uniform_pass']
            ok(cover['draws'] == 122 and cover['unique_clips_seen'] == 120
               and cover['repeat_visits'] == 2 and cover['full_pass_complete'], 'Full original120 coverage plus honest whole14 tailrepeats')
            ok(sum(value['draws'] for value in final['replay'].values()) == 8 * 8 + 7 * 2,
               'Replay counts obey old8perupdate thennew2, not new2 times globalstep')
            final_copy = fresh()
            final_copy.migrate_closed_state(closed, prior.identity)
            final_copy.load_state_dict(final)
            ok(final_copy.state_dict() == final, 'Second-cycle tail and segment layout exact-restorable')
            bad(new.next_batches, 'No implicit extension after remaining fullpass boundary')
            bad(lambda: fresh().load_state_dict(saved), 'Own exact requires original CLOSED boundary first')
            bad(lambda: fresh().migrate_closed_state(original, domain.identity), 'Legacy domain migration cannot masquerade as uniform continuation')
            for key in ('updates', 'replay_language_counts', 'domain_draws', 'source_group_draws', 'quota_segments', 'quota_transition'):
                changed = deepcopy(saved)
                if key == 'updates':
                    changed[key] += 1
                elif key == 'quota_segments':
                    changed[key][1]['TR_per_update'] = 8
                elif key == 'quota_transition':
                    changed[key]['prior_uniform_draws'] += 1
                else:
                    changed[key][next(iter(changed[key]))] += 1
                target = fresh()
                target.migrate_closed_state(closed, prior.identity)
                bad(lambda t=target, s=changed: t.load_state_dict(s), 'Reject corrupted own ' + key)
            changed = deepcopy(closed)
            changed['turkish_streams']['uniform']['order'][0] = changed['turkish_streams']['uniform']['order'][1]
            bad(lambda: fresh().migrate_closed_state(changed, prior.identity), 'Source duplicated order rejected')
            changed = deepcopy(closed)
            changed['identity']['indexed_TR_fingerprint'] = 'f' * 64
            bad(lambda: fresh().migrate_closed_state(changed, changed['identity']), 'Changed catalog/labels identity rejected')
            complete = old()
            for _ in range(15):
                complete.next_batches()
            bad(lambda: fresh().migrate_closed_state(complete.state_dict(), complete.identity), 'Completed first pass cannot be silently restarted')
            ok('torch' not in sys.modules and 'trackio' not in sys.modules, 'No ML/GPU/mirror imports')
        finally:
            indexed.close()
    report = {'schema': 1, 'kind': 'inactive_TSC_uniform_majority_sampler_tiny_stdlib_verification_v1',
        'passed': True, 'assertions': len(checks), 'rejections': len(rejections), 'checks': checks,
        'sources': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in
            ('tsc_uniform_majority_sampler_v1.py', 'tsc_uniform_full_sampler_v1.py', 'lazy_training_rows_v1.py')},
        'scope': 'Actual indexed disk carrier/native BucketCycler AST and sampler on120 synthetic TR plus24 replay rows. No real corpus/receipt/checkpoint/optimizer/Torch/GPU admission.',
        'actual_source_admitted': False, 'actual_207_Adam_or_GPU_restore_measured': False}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'rejections': len(rejections),
        'source_sha256': report['sources']['tsc_uniform_majority_sampler_v1.py'],
        'proof': str(args.output), 'proof_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'TSC_uniform_majority_sampler_tiny_stdlib_verification_v1.json')
    main(parser.parse_args())
