"""Conditional capacity successor with explicit directory-entry growth reserve."""
from copy import deepcopy
from bpe_full_run_capacity_v1 import allocation, derive as base_derive, integer, require, small_metadata


def derive(*, population, geometry, cleaner, replay, limits, disk):
    result = base_derive(population=population, geometry=geometry, cleaner=cleaner,
        replay=replay, limits=limits, disk=disk)
    # Directory base clusters do NOT bound growth of a flat directory's filename
    # entries. Reserve this separately; caller/source must also bound filenames,
    # per-publication file counts and reused transient directory trees.
    growth = integer(limits['maximum_directory_entry_growth_bytes'], 1)
    minimum = result['maximum_publications']*32*1024
    require(growth >= minimum, 'Separate directory-entry reserve at least32KiB/publication')
    rounded = allocation(growth, disk['allocation_unit_bytes'])
    result['kind'] = 'ASR18_conditional_full_run_cadence_capacity_arithmetic_v2'
    result['directory_base_allocation_bytes'] = result['directory_allocation_bytes']
    result['directory_entry_growth_bytes'] = rounded
    result['directory_entry_growth_minimum_conditional_bytes'] = minimum
    result['directory_allocation_bytes'] += rounded
    result['whole_required_bytes_including_reserve'] += rounded
    result['conditional_margin_bytes'] -= rounded
    result['conditional_arithmetic_fits'] = result['conditional_margin_bytes'] >= 0
    result['required_operational_proofs'].append(
        'Bounded filename entries/file counts and owned transient reuse keep directory growth within its separate cap.')
    return result


def from_pinned_metadata(*, inputs, limits, disk):
    require(set(inputs) == {'population', 'geometry', 'cleaner', 'replay'}, 'Four exact metadata authorities')
    values = {name: small_metadata(pin) for name, pin in inputs.items()}
    result = derive(**values, limits=limits, disk=disk)
    result['metadata_inputs'] = deepcopy(inputs)
    return result
