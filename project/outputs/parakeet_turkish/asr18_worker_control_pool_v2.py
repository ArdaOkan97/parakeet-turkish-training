"""Explicit operational reservations in an unchanged original control pool.

Only the new reservation parent changes.  The creator/source, callback authority,
registry, cumulative counters, archive rules and allocation limits stay v1.
This module cannot create a pool, launch a process or adopt an existing folder.
"""
from copy import deepcopy
from pathlib import Path
import re

from asr18_worker_control_pool_v1 import (
    WorkerControlPool as OriginalWorkerControlPool, CAPS, ROOT,
    bound, pin, require, safe, same,
)
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility


class WorkerControlPool(OriginalWorkerControlPool):
    def __init__(self, control, *, identity, policy, authority_source, guard,
                 metadata_phase, admit_control, operational_compatibility,
                 create=False):
        require(create is False, 'Operational continuation reopens the exact original pool only')
        review = checked_compatibility(operational_compatibility, identity, guard)
        require(same(review['operational_sources']['worker_control_pool'], pin(__file__, guard))
                and same(authority_source, review['legacy_guard_authority_source']),
                'Explicit ROOT-reviewed pool successor and unchanged original callback authority')
        self.operational_compatibility = deepcopy(operational_compatibility)
        self.operational_authority_source = deepcopy(review['operational_sources']['parent'])
        # The inherited constructor verifies the immutable original v1 owner,
        # including its component_source and callback authority.  No rewrite.
        super().__init__(control, identity=identity, policy=policy,
                         authority_source=authority_source, guard=guard,
                         metadata_phase=metadata_phase, admit_control=admit_control,
                         create=False)

    def reserve(self, folder, *, configuration, worker_source, parent_source):
        review = checked_compatibility(self.operational_compatibility, self.identity, self.guard)
        require(same(review['operational_sources']['worker_control_pool'], pin(__file__, self.guard))
                and same(review['legacy_guard_authority_source'], self.authority_source)
                and same(review['operational_sources']['parent'], self.operational_authority_source)
                and same(parent_source, self.operational_authority_source),
                'Actual reservation parent is the exact ROOT-reviewed operational successor')
        self._state = self._load()
        require(self._verified_protection, 'Fresh complete protection verification in each supervisor')
        path = safe(folder)
        require(not path.exists() and path.parent.is_dir() and len(self._state['live']) < CAPS['live_slots']
                and re.fullmatch(r'(?:native|recovery)_[0-9]{6}|DEV_[A-Za-z0-9_]{1,80}', path.name),
                'Fresh bounded worker reservation before any Popen/directory creation')
        require(path.parent.is_relative_to(ROOT / 'analysis') or path.parent.is_relative_to(ROOT / 'runs'),
                'Worker invocation root only in reviewed analysis/runs domain')
        config_path = bound(configuration, self.guard, CAPS['member_bytes'])
        require(config_path.is_relative_to(self.control) and not config_path.is_relative_to(self.folder),
                'Exact paired immutable owned worker configuration')
        bound(worker_source, self.guard, 262144); bound(parent_source, self.guard, 262144)
        require(str(path) not in {x['folder'] for x in self._state['live']}, 'No live worker directory reuse')
        item = {'folder': str(path), 'configuration': deepcopy(configuration),
                'worker_source': deepcopy(worker_source), 'parent_source': deepcopy(parent_source),
                'reservation': self._state['reservation_count'] + 1}
        state = deepcopy(self._state); state['live'].append(item); state['reservation_count'] += 1
        if str(path.parent) not in state['owner_folders']:
            require(len(state['owner_folders']) < CAPS['owner_folders'], 'Bounded persistent invocation-root count')
            state['owner_folders'].append(str(path.parent))
        self._save(state)
        return deepcopy(item)
