"""Tiny actual indexed/native queue extension test; no actual state/ML."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'outputs/parakeet_turkish'
sys.path[:0] = [str(HERE), str(ROOT/'work/research')]
from check_TSC_uniform_majority_sampler_v1 import (
    row, CountRows, cycler, prepare_indexed_rows, TSCDomainBalancedBucketSampler,
    TSCUniformFullSampler, TSCUniformMajoritySampler, canonical,
)
from tsc_uniform_majority_ptfocus_sampler_v1 import TSCUniformMajorityPTFocusSampler
import tsc_uniform_encoder_schedule_sampler_v1 as io


def main():
    checks = 0; rejected = 0
    def ok(value):
        nonlocal checks
        assert value; checks += 1
    def bad(call):
        nonlocal checks, rejected
        try:
            call()
        except ValueError:
            checks += 1; rejected += 1; return
        raise AssertionError('Corrupt state unexpectedly accepted')
    tr = [row('F'+str(i), group='fleurs') for i in range(36)] + [row('C'+str(i)) for i in range(24)] + [row('T'+str(i), group='tsc') for i in range(60)]
    replay = [row(lang+str(i), lang, 'fleurs') for lang in ('en','es','it','fr','de') for i in range(4)]
    pt = [row('pt'+str(i), 'pt', 'fleurs') for i in range(4)]
    kwargs = dict(seed=11, replay_seed=22, batch_size=8, accumulation=2, bucket_size=8,
                  replay_fraction=.5, fleurs_fraction=.3, cycler_factory=cycler, cycler_source_sha256='2'*64)
    domain = TSCDomainBalancedBucketSampler(tr, replay, sampling_mode='domain30', **kwargs)
    for _ in range(5): domain.next_batches()
    with tempfile.TemporaryDirectory(prefix='IO_sampler_tiny_', dir=ROOT/'work') as folder:
        indexed = prepare_indexed_rows(iter(tr), Path(folder)/'rows', expected_count=120,
            provenance={'synthetic_fixture_only':True, 'training_admission_complete':True,
                        'all_rows_verified':True, 'verified_clips':120})
        catalog = CountRows(indexed)
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
            source = focus.state_dict(); source_sha = canonical(source)
            def fresh():
                target = io.TSCUniformEncoderScheduleSampler(catalog, replay, **params)
                target.source_prefix = deepcopy(source['source_prefix'])
                target.quota_transition = deepcopy(source['quota_transition'])
                target.PT_focus_transition = deepcopy(source['PT_focus_transition'])
                target.parent_majority_identity = deepcopy(focus.parent_majority_identity)
                target.identity = deepcopy(source['identity'])
                target.load_state_dict(source)
                return target
            bad(fresh)  # Unmodified actual policy requires25109, not toy15.
            # Explicit synthetic clock-policy substitution only; all indexed
            # queues, frozen typed loaders/draw logic and RNG are real stdlib.
            io.SOURCE_END_STEP = 15; io.TARGET_POOL_CLIPS = 120; io.MATCHED_UPDATES = 9
            new = fresh()
            bad(new.next_batches)
            context = {'source_step':15, 'encoder_schedule_updates':9, 'configured_end_step':24,
                       'parent_identity':{'domain_sampling_identity':source['identity']},
                       'source_checkpoint':{'path':'synthetic_only', 'sha256':'a'*64, 'bytes':1}}
            before_reads = catalog.reads; new.begin_matched_cycle(context)
            ok(catalog.reads == before_reads)
            bad(lambda: new.begin_matched_cycle(context))
            initial = new.state_dict()
            ok(new.configured_end_step == 24 and new.original_full_pass_end_step == 15)
            ok(initial['quota_transition'] == source['quota_transition'] and initial['PT_focus_transition'] == source['PT_focus_transition'])
            ok(all(initial[k] == source[k] for k in source if k not in ('identity','kind')))
            ok(canonical(source) == source_sha)

            # Frozen parent draws as a literal future-batch oracle. Only its
            # temporary bound is widened; no queues/RNG/shuffle are reset.
            seeded = fresh()
            oracle = object.__new__(TSCUniformMajorityPTFocusSampler)
            oracle.__dict__ = seeded.__dict__.copy()
            oracle.quota_transition = deepcopy(source['quota_transition'])
            oracle.quota_transition['configured_end_step'] += 9
            oracle.quota_transition['remaining_update_budget'] += 9
            def ids(batches): return [[r['source_id'] for r in batch] for batch in batches]
            for _ in range(3): ok(ids(new.next_batches()) == ids(oracle.next_batches()))
            saved = new.state_dict()
            restored = fresh(); restored.begin_matched_cycle(context); restored.load_state_dict(saved)
            ok(restored.state_dict() == saved)
            for _ in range(6):
                a = ids(new.next_batches()); b = ids(restored.next_batches()); c = ids(oracle.next_batches())
                assert a == b == c
            checks += 1
            final = new.state_dict()
            ok(final['updates'] == 24 and final['matched_cycle_transition']['source_step'] == 15)
            ok(final['turkish_streams']['uniform']['draws'] == source['turkish_streams']['uniform']['draws']+126)
            ok(new.coverage()['matched_cycle']['new_unique_first_pass_clips'] == 0)
            ok(new.coverage()['uniform_pass']['full_pass_complete'])
            restored.load_state_dict(final); ok(restored.state_dict() == final)
            bad(new.next_batches)
            for field in ('matched_cycle_transition','quota_transition','quota_segments','updates'):
                corrupt = deepcopy(saved)
                if field == 'updates': corrupt[field] += 1
                elif field == 'quota_segments': corrupt[field][-1]['TR_draws'] += 1
                else: corrupt[field]['configured_end_step' if field == 'quota_transition' else 'updates'] += 1
                bad(lambda value=corrupt: restored.load_state_dict(value))
            ok((253798-2+13)//14 == 18129 and 18129*14 == 253806 and 253806-253796 == 10)
            ok(new.coverage()['matched_cycle']['cycle_unique'] == 120 and new.coverage()['matched_cycle']['tail_draws'] == 8)
            ok('torch' not in sys.modules and 'trackio' not in sys.modules)
        finally:
            io.SOURCE_END_STEP = 25109; io.TARGET_POOL_CLIPS = 253798; io.MATCHED_UPDATES = 18129
            indexed.close()
    path = HERE/'TSC_uniform_encoder_schedule_sampler_tiny_stdlib_verification_v1.json'
    report = {'schema':1, 'kind':'matched_encoder_next_cycle_sampler_stdlib_verification_v1',
              'passed':True, 'assertions':checks, 'rejections':rejected,
              'source_sha256':hashlib.sha256((HERE/'tsc_uniform_encoder_schedule_sampler_v1.py').read_bytes()).hexdigest(),
              'scope':'Actual indexed120 synthetic TR/native replay cyclers and typed frozen loaders. Actual25109 policy rejection tested, then source clock15/pool120/derived9updates explicitly substituted only for tiny next-cycle draw/restore math. No actual source/PT/model/GPU admission.',
              'same_next_cycle_future_batches_vs_frozen_parent':True, 'new_firstpass_unique_clips_claimed':False}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({**report,'proof':str(path),'proof_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == '__main__': main()
