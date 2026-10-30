"""Inert source review + tiny fixtures; never read manifests/models/audio/evals."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
REPORT = OUT / 'turkish_domain_sampler_integration_readiness_v1.json'
sys.path.insert(0, str(OUT))
import turkish_domain_sampler_v1 as domain_module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if REPORT.exists():
        raise FileExistsError('Prior readiness artifact must remain unchanged')
    trainer = OUT / 'train_lora_batched_data_v11.py'
    preparation_path = OUT / 'turkish_domain_sampler_preparation_v1.json'
    proof = json.loads(preparation_path.read_text(encoding='utf-8'))
    if (not proof.get('passed') or proof['module_sha256'] != sha(domain_module.__file__)
            or proof['cycler_source_sha256'] != sha(trainer)):
        raise ValueError('Existing completed preparation does not bind current sources')
    tree = ast.parse(trainer.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if getattr(node, 'name', None)
             in {'BucketCycler', 'BalancedBucketSampler', 'ids_sha'}]
    scope = {'hashlib': hashlib, 'math': math, 'random': random,
             'LANGUAGES': ['en', 'es', 'it', 'fr', 'de', 'tr']}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(trainer), 'exec'), scope)
    def tr_row(domain, index):
        return {'source_id': f'{domain}{index}', 'language': 'tr',
                'encoder_frames': 12 + index, 'target_tokens': 2 + index % 9,
                'duration': 11.0 + index / 10 if domain == 'fleurs' else 2.0 + index / 10,
                'dataset': 'google/fleurs' if domain == 'fleurs' else 'ysdede/commonvoice_17_tr_fixed',
                'dataset_revision': domain_module.FLEURS_REVISION if domain == 'fleurs' else domain_module.CV_REVISION,
                'dataset_config': 'tr_tr' if domain == 'fleurs' else 'default', 'dataset_split': 'train'}
    tr = [tr_row('fleurs', i) for i in range(7)] + [tr_row('cv', i) for i in range(23)]
    replay = [{'source_id': f'{lang}{i}', 'language': lang, 'encoder_frames': 15 + i,
               'target_tokens': 5 + i, 'duration': 9.0 + i / 10}
              for lang in domain_module.LANGUAGES for i in range(5)]
    kwargs = dict(seed=20261010, batch_size=8, accumulation=2, replay_fraction=.5, bucket_size=8)
    def domain(**overrides):
        return domain_module.DomainBalancedBucketSampler(tr, replay, **kwargs,
            fleurs_fraction=overrides.get('fleurs_fraction', .3), cycler_factory=scope['BucketCycler'],
            cycler_source_sha256=sha(trainer))
    checks = {}
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks[name] = True
    global_rng = random.getstate()
    weighted = domain()
    uniform = scope['BalancedBucketSampler'](tr, replay, **kwargs)
    first_quotas, weighted_replay, uniform_replay = [], [], []
    for update in range(10):
        batches = weighted.next_batches()
        check(f'B8x2_TR8_replay8_update_{update}', len(batches) == 2 and all(len(batch) == 8 for batch in batches)
              and sum(row['language'] == 'tr' for batch in batches for row in batch) == 8)
        first_quotas.append(sum(row['dataset'] == 'google/fleurs'
            for batch in batches for row in batch if row['language'] == 'tr'))
        weighted_replay.extend(row['source_id'] for batch in batches for row in batch if row['language'] != 'tr')
        uniform_replay.extend(row['source_id'] for batch in uniform.next_batches() for row in batch if row['language'] != 'tr')
    check('first5_FLEURS_quotas_2_2_3_2_3', first_quotas[:5] == [2, 2, 3, 2, 3])
    check('exact_30_percent_of80_TR_draws', weighted.coverage()['domain']['fleurs']['draws'] == 24)
    check('50_percent_replay80_draws', sum(s.draws for s in weighted.replay.values()) == 80)
    check('replay_sequence_is_not_uniform_reference_sequence', weighted_replay != uniform_replay)
    check('no_global_Python_RNG_change', random.getstate() == global_rng)
    checkpoint = copy.deepcopy(weighted.state_dict())
    restored = domain(); restored.load_state_dict(checkpoint)
    check('state_restore_exact', restored.state_dict() == checkpoint)
    for _ in range(3):
        expected = [[row['source_id'] for row in batch] for batch in weighted.next_batches()]
        actual = [[row['source_id'] for row in batch] for batch in restored.next_batches()]
        check('three_resumed_updates_identical_' + str(_), expected == actual)
    check('resumed_full_sampler_state_exact', weighted.state_dict() == restored.state_dict())
    try:
        domain(fleurs_fraction=.4).load_state_dict(checkpoint)
    except ValueError:
        check('changed_fraction_rejected_on_exact_resume', True)
    else:
        raise AssertionError('Changed fraction was accepted')
    check('direct_trainer_swap_missing_tr_and_turkish_epoch',
          not hasattr(restored, 'tr') and not hasattr(restored, 'turkish_epoch'))
    f = proof['pools']['fleurs']; c = proof['pools']['cv']; total = f['clips'] + c['clips']
    def required_updates(epochs):
        needed_f, needed_c = math.ceil(epochs * f['clips']), math.ceil(epochs * c['clips'])
        updates = math.ceil(max(needed_f / .3, needed_c / .7) / 8)
        while 8 * updates * 3 // 10 < needed_f or 8 * updates - 8 * updates * 3 // 10 < needed_c:
            updates += 1
        return updates
    required = required_updates(1)
    uniform_steps = math.ceil(total / 8)
    domain_draws_at_uniform_steps = {'fleurs': 8 * uniform_steps * 3 // 10,
                                   'cv': 8 * uniform_steps - 8 * uniform_steps * 3 // 10}
    check('receipt_one_domain_pass_budget_recomputed', required == proof['B8x2_TR8_updates_for_one_domain_coverage_pass'])
    report = {
        'schema': 1, 'kind': 'inert_Turkish_domain_sampler_integration_readiness_v1',
        'status': 'source_and_tiny_fixture_review_complete_not_integrated', 'passed': True,
        'CPU_only': True, 'Torch_NeMo_model_audio_imports': False, 'GPU_launch': False,
        'manifests_checkpoints_DEV_TEST_read': False, 'active_or_frozen_sources_modified': False,
        'source_bindings': {'trainer_path': str(trainer), 'trainer_sha256': sha(trainer),
            'sampler_path': str(Path(domain_module.__file__).resolve()), 'sampler_sha256': sha(domain_module.__file__),
            'prior_preparation_path': str(preparation_path), 'prior_preparation_sha256': sha(preparation_path),
            'report_builder_sha256': sha(__file__)},
        'prior_receipt_reused_without_full_manifest_scan': {'passed': proof['passed'], 'check_count': proof['check_count'],
            'accepted_manifest_sha256': proof['manifest']['sha256'], 'accepted_TR_clips': total,
            'domain_pools': proof['pools'], 'prior_live_history_snapshot_is_historical_NOT_current_status': True},
        'sampling_semantics': {'effective_batch': 16, 'physical_batch': 8, 'accumulation': 2,
            'TR_clips_per_update': 8, 'replay_clips_per_update': 8, 'TR_domain_fraction': {'fleurs': .3, 'cv': .7},
            'quota': 'floor(total_TR_draws*3/10) FLEURS; remainder CV, incremental quota error less than1 clip',
            'FLEURS_clip_counts_first5_updates': first_quotas[:5],
            'domain_cycles': 'Unique-source shuffled cycles with existing128-row log-T/log-U reservoirs inside each domain',
            'TR_packing': 'After choosing2 or3 FLEURS plus6 or5 CV, sort those8 selected TR rows by T/U and pack one B8 TR microbatch',
            'replay_distribution': 'Uniform random language per replay clip; genuine original teacher caches and existing per-language bucket cyclers',
            'fractions_are_clip_counts_not_audio_weights': True,
            'FLEURS_audio_share_of_TR_expectation_from_prior_mean_durations': proof['expected_FLEURS_share_of_Turkish_audio_from_mean_durations'],
            'replay_exact_source_sequence_matches_uniform_sampler': False,
            'reason_replay_sequence_changes': 'Shared local RNG consumes different domain shuffles, changing replay language choices and row order',
            'padding_and_throughput_of_30percent_domain_recipe_unmeasured': True},
        'draws_versus_pass_accounting': {
            'uniform_one_pool_draw_budget_updates': uniform_steps,
            'weighted_one_domain_coverage_pass_updates': required,
            'at_uniform_draw_budget_domain_draws': domain_draws_at_uniform_steps,
            'at_uniform_draw_budget_maximum_CV_unique_fraction': domain_draws_at_uniform_steps['cv'] / c['clips'],
            'at_domain_pass_total_TR_draws': required * 8,
            'at_domain_pass_FLEURS_draws': required * 8 * 3 // 10,
            'at_domain_pass_CV_draws': required * 8 - required * 8 * 3 // 10,
            'full_pool_pass_definition': 'Every source seen in both domains; draws/full_pool_size alone is not a full pass',
            'required_history': ['domain draws', 'domain unique seen', 'domain repeat visits', 'domain completed cycles',
                'domain audio seconds', 'total TR draws', 'draw-equivalent epochs', 'minimum domain exposure epochs',
                'all TR sources seen', 'replay language counts', 'TR and combined audio minutes per train wall minute'],
            'frame_U_and_audio_metadata': 'Reuse accepted frozen labels/PCM/T/teacher-cache metadata unchanged'},
        'concrete_integration_blockers': [
            {'problem': 'Not a drop-in trainer11 sampler: no .tr or .turkish_epoch',
             'impact': 'Resume clip check, epoch/DEV/plateau clocks, history unique/draw counters and final selection would fail or misreport',
             'minimum_fix': 'Use explicit exposure accessors and saved sampler kind; sum domain draws/unique counts, choose minimum-domain exposure for epoch clock and require all sources seen for pass claims'},
            {'problem': 'Trainer11 epoch_steps assumes uniform len(TR)/TRclips_per_update',
             'impact': 'A configured1epoch stops after9361 updates while about28percent of CV is still unseen',
             'minimum_fix': 'For domain mode use updates_for_minimum_domain_exposure(args.epochs); retain absolute --steps/time caps and report partial coverage truthfully'},
            {'problem': 'Uniform sampler state is incompatible with two domain streams',
             'impact': 'Old RNG/order/cursors/seen cannot honestly be loaded as exact domain continuation',
             'minimum_fix': 'Start a NEW warmstart arm from the latest closed qualifying weights with declared fresh optimizer/sampler; later exact resumes restore this domain state and identical recipe identity'},
            {'problem': 'Same seed does not preserve the exact uniform replay sequence',
             'impact': 'Existing sampler preserves replay quota/distribution only, not a strict paired replay-order control',
             'minimum_fix': 'Declare this nuisance for a first exploratory arm; for strict paired testing use independent identical replay RNG/plan in both NEW uniform and domain arms'}
        ],
        'smallest_future_integration': [
            'Add one explicit uniform/domain sampling switch plus .3 FLEURS fraction in a future isolated trainer; default remains uniform.',
            'Supply identical BucketCycler code, bind sampler and cycler source SHA, fraction3/10, bucket128, pool fingerprints and seed in run/checkpoint identity.',
            'Keep every audio/text/tokenizer/target/cache identity, model/rank/weights, B8x2, LR/schedule, native loss and KD/new-character multipliers fixed; only domain sampling and honest exposure clock change.',
            'Replace direct sampler.tr references with an explicit exposure record; add domain coverage fields rather than silently relabel old nominal draw epochs as passes.',
            'Use minimum domain exposure for epoch budget/cadence and completed domain-cycle advances for full-pass plateau counting; keep short-update DEV checks and all5 retention gates.',
            'Save/restore the module state unchanged: shared RNG, completed updates, both domain orders/cursors/seen/draws, replay orders/cursors/counts and domain audio totals.',
            'Use a new run namespace and latest CLOSED DEV-qualified warmstart; no migration of current uniform Adam/sampler as if mathematically identical.',
            'Measure actual domain/total unique coverage, repeats, TR audio exposure, padding and wall time. Comparisons at equal update counts change audio exposure; report both update and TR-audio budgets.',
            'Do not combine the first domain arm with BPE512, rank16, LR/KD or replay-fraction changes. Any strict paired control should share a separately pinned replay plan.'
        ],
        'current_action': 'Continue the live uniform-pass recipe; root reported step2880 / DEV2750 TR59.2727percent with all5 retention gates passing. This review does not read or verify live evaluation files.',
        'future_activation_pending': True, 'new_runtime_chain_created': False,
        'tiny_fixture_evidence': {'pool_clips': {'fleurs': 7, 'cv': 23, 'replay_each_language': 5},
            'matched_initial_update_count': 10, 'resumed_followup_updates': 3,
            'check_count': len(checks), 'checks': checks,
            'weighted_vs_uniform_replay_sequence_equal': weighted_replay == uniform_replay,
            'sampler_restored_state_equal': weighted.state_dict() == restored.state_dict()},
        'limitations': ['Readiness is source/stdlib sampler evidence, not runtime integration or measured quality/memory/throughput.',
                        'Existing preparation receipt supplies complete accepted-domain counts and eligibility; no full manifest scan was repeated.',
                        'Prior preparation history snapshots are historical and must not be interpreted as current coverage.']
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(REPORT), 'passed': True, 'check_count': len(checks),
                      'uniform_budget_updates': uniform_steps, 'domain_coverage_updates': required,
                      'CV_unique_fraction_at_uniform_budget_upper_bound': domain_draws_at_uniform_steps['cv'] / c['clips'],
                      'replay_sequence_equal': weighted_replay == uniform_replay}), flush=True)


if __name__ == '__main__':
    main()
