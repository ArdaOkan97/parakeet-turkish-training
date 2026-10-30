"""INACTIVE source-bound capture at the sole idle refresh, never per-update.

Two calls of the ORIGINAL bounded fingerprint add two explicit normal-case CUDA
synchronizations plus <=4MiB tensor transfer slices. This is not a no-payload-read
or version-only proof. No whole CPU model copy or payload output is introduced.
Only scalar/hash trees and identity/storage/version metadata survive each call.
"""
from copy import deepcopy


def tensor_identity(value):
    return {'id': id(value), 'data_ptr': value.data_ptr(), 'version': value._version,
            'shape': list(value.shape), 'stride': list(value.stride()),
            'dtype': str(value.dtype), 'device': str(value.device),
            'storage_offset': value.storage_offset(), 'requires_grad': value.requires_grad}


def capture_idle_state(*, optimizer, torch, guard, schedule, context, journal,
                       lifecycle, periodic, queued, latest, completed):
    from bpe_broad_audio_v1 import record, require
    from bpe_broad_production_state_v1 import digest
    from asr18_live_state_fingerprint_v1 import live_state_fingerprint
    from asr18_BPE9223_source_context_v2 import full_run_next_draw_packet
    guard()
    require(not optimizer.poisoned and not schedule.poisoned
            and len(optimizer.parameters) == 207 and len(optimizer.optimizers) == 3,
            'Actual unpoisoned scientific owners at idle refresh')
    require(type(queued) is tuple and len(queued) == 2 and record(queued[1]['path']) == queued[1],
            'Original queued tuple, exact original object/content and saved pin')
    parameters = {}
    for name, value in sorted(optimizer.parameters.items()):
        require(value.grad is None, 'All207 real gradients absent at idle barrier')
        parameters[name] = tensor_identity(value)
    partitions = []
    count = 0
    for owner in optimizer.optimizers:
        groups = []
        for group in owner.param_groups:
            states = []
            for parameter in group['params']:
                state = owner.state[parameter]
                require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'All three tensors of each Adam slot')
                require(parameter.grad is None, 'Adam Parameter views have no live gradient')
                states.append({'parameter': tensor_identity(parameter), 'state_id': id(state),
                               'tensors': {k: tensor_identity(state[k]) for k in ('step', 'exp_avg', 'exp_avg_sq')}})
                count += 1
            groups.append({'policy': deepcopy({k: v for k, v in group.items() if k != 'params'}), 'states': states})
        partitions.append({'owner_id': id(owner), 'state_owner_id': id(owner.state), 'groups': groups})
    require(count == 215 and len(parameters) == 207, 'Exact original207/threeAdam215 slot identity coverage')
    fingerprint = live_state_fingerprint(optimizer, torch=torch, guard=guard)
    require(len(fingerprint['state']['weights']) == 207
            and sum(len(v['state']) for v in fingerprint['state']['Adam_partitions']) == 215,
            'Unmodified original full semantic fingerprint, not an identity-only replacement')
    schedule_state = schedule.state_dict()
    require(latest['full_state_verified'] is True and latest['committed_updates'] == optimizer.updates
            and schedule.committed_updates() == optimizer.updates,
            'Same fully persisted idle native/sampler/checkpoint frontier')
    return {'independent_original_live_fingerprint': fingerprint,
            'parameters': parameters, 'Adam_owners_and_215_slots': partitions,
            'optimizer_owner_id': id(optimizer), 'schedule_owner_id': id(schedule),
            'context_owner_id': id(context), 'resolver_owner_id': id(context.resolver),
            'parent_metadata_lock_handles': list(context.resolver.metadata_locks.handles),
            'journal_owner_id': id(journal), 'journal': journal.binding(),
            'schedule': schedule_state,
            'next_draw': full_run_next_draw_packet(schedule, context, optimizer_updates=optimizer.updates, guard=guard),
            'lifecycle': deepcopy(lifecycle), 'periodic': deepcopy(periodic), 'completed_boundaries': completed,
            'queued_tuple_id': id(queued), 'queued_plan_id': id(queued[0]), 'queued_pin_id': id(queued[1]),
            'queued_plan_sha256': digest(queued[0]), 'queued_pin': deepcopy(queued[1]),
            'latest': deepcopy(latest), 'clocks': [c + optimizer.updates for c in optimizer.base_clocks],
            'full_semantic_byte_equality_uses_original_routine_not_versions_alone': True}


def compare_idle_state(before, after):
    from asr18_live_state_fingerprint_v1 import require_exact_live_restore
    require_exact_live_restore(before['independent_original_live_fingerprint'],
                               after['independent_original_live_fingerprint'])
    if before != after:
        raise ValueError('Actual identities/storage/versions/gradients/RNG/samplers/journal/queued/lifecycle changed')
