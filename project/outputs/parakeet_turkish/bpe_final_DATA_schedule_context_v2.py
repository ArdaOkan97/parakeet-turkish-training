"""Concrete original-source schedule and actual-checkpoint next-window planner.

This module never loads an ASR model or advances an authoritative draw. Window2
is derived from the actual saved sampler, independently bound to a coherent
native boundary; a virtual window1 planning tail is never an input substitute.
"""
from copy import deepcopy
import json
from pathlib import Path
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import CarriedReplay, digest
from bpe_training_schedule_v1 import TrainingSchedule, WEAK
from bpe_replay_bucket_no_runtime_v1 import BucketCycler
from bpe_original_schedule_snapshot_v1 import load_original_snapshot, python_random_state
from bpe_replay_cache_labels_v1 import full_replay_labels
from bpe_final_DATA_window_planner_v4 import FinalBroadRows, feature_upper_bound
from bpe_production_window_coordinator_v4 import make_plan
from bpe_streaming_checkpoint_semantics_v2 import select, decode_scalar_tree, verify_saved_metadata
from fastmss_phase_aware_plan_bridge_v3 import validate_DATA, actual0, read_bound


def normalize_schedule_state(value):
    result = deepcopy(value)
    for key in ('replay_rng', 'Portuguese_rng'):
        result['carried_replay'][key] = python_random_state(result['carried_replay'][key])
    return result


class ScheduleContext:
    def __init__(self, arguments, *, guard):
        self.arguments = deepcopy(arguments); self.guard = guard; self.closed = False
        self.validated = validate_DATA(arguments['DATA_evidence'], guard)
        actual0(arguments['snapshot_actual0'], arguments['snapshot_report'], guard, owner='/root/memory_diagnosis')
        report = read_bound(arguments['snapshot_report'], guard)
        require(report['passed'] is True and report['snapshot'] == arguments['snapshot_binding']
                and report['all_original_RNG_queues_counts_exact'] is True
                and report['actual_optimizer_updates'] == report['draws_consumed'] == 0,
                'Actual untouched original schedule extraction proof required')
        self.snapshot = load_original_snapshot(arguments['snapshot_binding'],
                expected_sources=arguments['expected_original_sources'], guard=guard)
        pool = arguments['replay_pool_binding']; cleaner = arguments['cleaner_metadata_binding']
        require(pool['sha256'] == '00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4'
                and record(pool['path']) == pool and cleaner['sha256'] == '4a1c2bf698e038f2ea7c886ccf14281e197a01ac77836c8134c9b5b4b010b99a'
                and record(cleaner['path']) == cleaner, 'Exact full original replay/cleaner metadata identities required')
        self.pool = json.loads(Path(pool['path']).read_text(encoding='utf-8'))['replay']
        self.cleaner = json.loads(Path(cleaner['path']).read_text(encoding='utf-8'))
        require(len(self.pool) == 1914 and self.cleaner['passed'] is True
                and self.cleaner['identity']['rows'] == 352768
                and self.cleaner['identity']['full_target_tokens'] == 11972886,
                'All original replay rows and full cleaner denominator required')
        self.identity = {'kind': 'ASR18_final_unfiltered_broad_DATA_identity_v1',
                        'population': arguments['DATA_evidence']['population'],
                        **{key: self.validated['bindings'][key] for key in
                           ('final_catalog', 'final_labels', 'declared_geometry_index', 'completed_source_inventory')}}
        self.resolver = FinalBroadRows(self.validated, guard=guard)
        self.replay_evidence = {}

    def new_schedule(self):
        require(not self.closed, 'Closed source context')
        return TrainingSchedule(broad_population=self.validated['retained_rows'], broad_identity=self.identity,
            broad_mean_U=self.validated['population_mean_U'], cleaner_population=352768,
            cleaner_identity=self.cleaner['identity'], cleaner_mean_U=11972886/352768,
            source_turkish_rng=self.snapshot['turkish_rng'], replay=CarriedReplay(self.pool, self.snapshot, BucketCycler))

    def resolve_identity(self, phase, ordinal):
        # The finite first<=40 caller must stop at a DEV/phase barrier. A future
        # cleaner/full production coordinator needs its separately bound reader.
        require(phase == WEAK, 'No implicit cleaner transition from this bounded broad prefix')
        row = self.resolver.row(ordinal)
        return {'ordinal': ordinal, 'source_id': row['source_id'], 'language': 'tr'}

    def resolve_rows(self, phase, draws, replay):
        require(phase == WEAK and not self.closed, 'Explicit bounded broad source context required')
        rows = [self.resolver.row(draw['ordinal']) for draw in draws]
        for original in replay:
            self.guard(); cache = record(original['cache_path'])
            labels, proof = full_replay_labels(original, cache, guard=self.guard)
            key = original['source_id']
            require(key not in self.replay_evidence or self.replay_evidence[key] == proof,
                    'Original full replay cache/target binding changed')
            self.replay_evidence[key] = proof
            rows.append({**original, 'augmentation_role': 'replay', 'clean_target_ids': labels,
                         'full_replay_label_binding': proof})
        return rows

    def plan(self, schedule, *, maximum_updates):
        require(schedule.phase == WEAK and schedule.committed_updates() <= 40,
                'Only a bounded original broad prefix; no exhausted diagnostic cursor')
        return make_plan(schedule, clone_schedule=self.new_schedule, resolve_identity=self.resolve_identity,
            resolve_rows=self.resolve_rows, feature_bytes_upper_bound=feature_upper_bound,
            finalized_population=self.arguments['DATA_evidence']['population'],
            verify_finalized_population=lambda pin: pin == self.arguments['DATA_evidence']['population'],
            maximum_updates=maximum_updates)

    def close(self):
        if not self.closed: self.resolver.close(); self.closed = True


def actual_saved_schedule(pointer_binding, *, guard):
    pointer = read_bound(pointer_binding, guard)
    require(pointer['kind'] == 'ASR18_production_window_completed_checkpoint_pointer_v2'
            and pointer['diagnostic_only'] is True and pointer['all_window_updates_committed'] is True
            and pointer['writer_returned_and_payload_verified'] is True,
            'Actual completed whole-window checkpoint pointer required')
    metadata = read_bound(pointer['checkpoint'], guard)
    state = decode_scalar_tree(select(metadata['tree'], 'sampler'))
    verify_saved_metadata(metadata, identity=pointer['identity'], optimizer_updates=pointer['committed_updates'],
                          sampler_state=state, journal_prefix=pointer['journal_prefix'], diagnostic_only=True)
    guard(); require(record(metadata['payload']['path']) == metadata['payload'] == pointer['payload'],
                     'Full coherent saved payload must match returned pointer')
    require(digest(state) == pointer['sampler_state_sha256'], 'Saved sampler differs from actual checkpoint pointer')
    return pointer, state


def prepare_following_plan(arguments, *, pointer_binding, native_boundary_binding,
                           verify_native_boundary, maximum_updates, guard):
    require(callable(verify_native_boundary) and verify_native_boundary(native_boundary_binding, pointer_binding) is True,
            'Actual closed or live-quiescent coherent source-bound native boundary required')
    pointer, state = actual_saved_schedule(pointer_binding, guard=guard)
    require(20 <= pointer['committed_updates'] <= 21 and type(maximum_updates) is int
            and maximum_updates == 1 and pointer['committed_updates']+maximum_updates <= 22,
            'Only actual20→21→22 one-update diagnostic continuation, no scientific reset')
    context = ScheduleContext(arguments, guard=guard)
    try:
        schedule = context.new_schedule(); schedule.load_state_dict(normalize_schedule_state(state))
        before = deepcopy(schedule.state_dict()); plan = context.plan(schedule, maximum_updates=maximum_updates)
        require(digest(before) == plan['initial_sampler_sha256'] == pointer['sampler_state_sha256']
                and schedule.state_dict() == before, 'Planning cannot advance actual restored frontiers')
        return {'kind': 'ASR18_actual_checkpoint21_22_diagnostic_window_preparation_v1', 'plan': plan,
                'initial_sampler': before, 'parent_pointer': pointer_binding,
                'actual_native_boundary': native_boundary_binding,
                'replay_metadata_label_proofs': context.replay_evidence,
                'actual_optimizer_updates_performed_by_preparation': 0,
                'actual_authoritative_draws_consumed_by_preparation': 0, 'production_enabled': False}
    finally: context.close()
