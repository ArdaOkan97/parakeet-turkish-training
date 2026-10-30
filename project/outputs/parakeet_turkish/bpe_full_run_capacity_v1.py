"""Pure, conditional full-run cadence/storage arithmetic; never admission.

Only small pinned metadata is read. No database, audio, payload, native work,
retirement, sampler or source mutation is performed. Operational packing,
emergency handling, archive caps and directory retirement need separate proof.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

E = Path(r'E:\gpu_training_scratchpad')


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'Exact nonboolean integer required')
    return value


def ceildiv(a, b):
    return (integer(a)+integer(b, 1)-1)//b


def allocation(size, unit):
    return ceildiv(integer(size), integer(unit, 1))*unit


def small_metadata(pin):
    """Hash the exact small metadata file, never any referenced bulk artifact."""
    require(type(pin) is dict and {'path', 'bytes', 'sha256'} <= set(pin)
            and set(pin) <= {'path', 'bytes', 'sha256', 'mtime_ns'}, 'Small metadata binding')
    path = Path(pin['path']).resolve()
    require(path.is_relative_to(E.resolve()) and not path.is_symlink(), 'Pinned metadata on E only')
    size = integer(pin['bytes'], 1)
    require(size <= 4*1024**2 and path.stat().st_size == size, 'Small bounded exact file size')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Metadata hash mismatch')
    value = json.loads(raw)
    require(type(value) is dict, 'Metadata object required')
    return value


def derive(*, population, geometry, cleaner, replay, limits, disk):
    """Upper bound publications IF the stated normal-window contract holds.

    Four worst-case weak updates must fit; a normal window cannot voluntarily
    stop earlier, except at a pass tail. Any coherent partial window spends an
    explicit emergency publication. This contract is an assumption, not proved
    by arithmetic, and must be enforced by the reviewed production caller.
    """
    require(population['complete'] is True and population['pass_count'] == 2
            and not population['quality_filters'] and not population['language_filters'],
            'Complete unfiltered two-pass population')
    weak = integer(population['retained_rows'], 1)
    require(population['label_rows'] == population['frontier'] == weak
            and population['ordinal_domain'] == {'min': 1, 'max': weak, 'contiguous_unique': True},
            'Final contiguous count/frontier')
    require(geometry['complete'] is True and geometry['population']['retained_rows'] == weak
            and geometry['population']['total_target_tokens'] == population['total_target_tokens']
            and geometry['unknown_invalid_counts']['retained_unresolved_rows'] == 0,
            'Same final declared geometry/count/tokens; no unresolved geometry')
    require(cleaner['kind'] == 'ASR18_cleaner_complete_scalar_conditional_resource_handoff_v1'
            and cleaner['full_scalar_metadata_and_conditional_arithmetic_actual0'] is True
            and cleaner['augmentation_bypassed'] is True
            and replay['kind'] == 'ASR18_whole_original_replay_labels_actual_readiness_v1'
            and replay['passed'] is True and replay['all_original_rows_retained'] is True,
            'Complete cleaner and original replay scalar metadata')
    clean = integer(cleaner['rows'], 1)
    require(integer(replay['rows'], 1) == 1914, 'Original complete replay population')
    gp = geometry['population']
    weak_frames = integer(gp['MAX_clean16k_samples'], 1)//160+1
    replay_frames = 8*integer(replay['MAX_T'], 1)+1
    clean_frames = 8*integer(cleaner['MAX_declared_T'], 1)+1
    # Full physical final zero-masked frame is retained. One FP32[frames,128]
    # feature plus one bool[frames] mask, with TWO <=4096-byte NPY headers.
    row_bytes = lambda frames: frames*(128*4+1)+2*4096
    weak_bytes = 14*row_bytes(weak_frames)+2*row_bytes(replay_frames)
    clean_bytes = 14*row_bytes(clean_frames)+2*row_bytes(replay_frames)
    cache = integer(limits['feature_cache_bytes'], 1)
    reserve = integer(limits['feature_metadata_reserve_bytes'], 1)
    require(cache == 512*1024**2 and reserve >= 16*1024**2 and reserve < cache,
            'Original 512MiB window with explicit metadata reserve')
    maximum = integer(limits['maximum_window_updates'], 1)
    require(maximum == 20, 'Reviewed finite window ceiling remains20')
    weak_min = min(maximum, (cache-reserve)//weak_bytes)
    clean_min = min(maximum, (cache-reserve)//clean_bytes)
    require(weak_min >= 1 and clean_min >= 1, 'Every complete update must fit; no row exclusion')
    weak_updates, clean_updates = ceildiv(weak, 14), ceildiv(clean, 14)
    normal = 2*ceildiv(weak_updates, weak_min)+ceildiv(clean_updates, clean_min)
    emergency = integer(limits['maximum_emergency_publications'])
    publications = normal+emergency
    cluster = integer(disk['allocation_unit_bytes'], 1)
    require(cluster & (cluster-1) == 0 and cluster <= 1024**2, 'Actual power-of-two allocation unit')
    free = integer(disk['free_bytes'])
    caps = limits['artifact_caps_bytes']
    require(set(caps) == {'payload', 'metadata', 'pointer', 'archive', 'window_auxiliary'},
            'Complete explicit artifact cap set')
    rounded = {name: allocation(integer(size, 1), cluster) for name, size in caps.items()}
    live = integer(limits['maximum_live_and_inflight_payloads'], 1)
    require(live >= 5, 'Current plus qualified winners and one in-flight payload')
    # Count directory allocations explicitly; zero-cost empty directories are
    # never assumed. A per-window directory scheme must supply its full count.
    retained_dirs = integer(limits['maximum_retained_owned_directories'], 1)
    transient_dirs = integer(limits['maximum_inflight_owned_directories'], 1)
    directories = (retained_dirs+transient_dirs)*cluster
    record_cap = integer(limits['durable_record_cap_bytes'], 41)
    require(record_cap <= 65536, 'Bounded framed durable record cap')
    extra_records = integer(limits['maximum_additional_durable_records'], 1)
    stream = allocation((publications+extra_records)*record_cap, cluster)
    index = allocation(integer(limits['maximum_owned_index_bytes'], 1), cluster)
    other = allocation(integer(limits['other_journal_log_control_growth_bytes'], 1), cluster)
    disk_reserve = integer(limits['E_reserve_bytes'], 100_000_000_000)
    retained = publications*(rounded['metadata']+rounded['pointer']+rounded['archive'])
    auxiliary = publications*rounded['window_auxiliary']
    payloads = live*rounded['payload']
    required = retained+auxiliary+payloads+directories+stream+index+other+disk_reserve
    return {'kind': 'ASR18_conditional_full_run_cadence_capacity_arithmetic_v1',
        'weak_rows': weak, 'cleaner_rows': clean, 'TR_per_update': 14, 'replay_per_update': 2,
        'weak_passes': 2, 'cleaner_passes': 1,
        'weak_updates_per_pass': weak_updates, 'cleaner_updates': clean_updates,
        'total_native_updates': 2*weak_updates+clean_updates,
        'physical_feature_frame_bounds': {'weak': weak_frames, 'replay': replay_frames, 'cleaner': clean_frames},
        'worst_complete_update_feature_bytes': {'weak': weak_bytes, 'cleaner': clean_bytes},
        'minimum_full_normal_window_updates': {'weak': weak_min, 'cleaner': clean_min},
        'maximum_normal_publications': normal, 'maximum_emergency_publications': emergency,
        'maximum_publications': publications, 'allocation_rounded_artifact_caps': rounded,
        'retained_metadata_pointer_archive_bytes': retained, 'retained_auxiliary_archive_bytes': auxiliary,
        'live_and_inflight_payload_bytes': payloads, 'directory_allocation_bytes': directories,
        'durable_stream_bytes': stream, 'owned_index_bytes': index,
        'other_journal_log_control_growth_bytes': other, 'E_reserve_bytes': disk_reserve,
        'whole_required_bytes_including_reserve': required,
        'observed_free_bytes': free, 'conditional_margin_bytes': free-required,
        'conditional_arithmetic_fits': required <= free,
        'disk_observation': deepcopy(disk), 'limits': deepcopy(limits),
        'required_operational_proofs': [
            'Greedy normal packing reaches the minimum above; neither broad-pass boundary is crossed.',
            'Coherent partial stops use the explicit emergency quota; no unlimited budget/resource retries.',
            'All file/archive/durable/index and directory caps enforced by actual future-owned sources.',
            'Lossless archive resolution preserves exact historical identities; current and protected files stay live.',
            'Full data geometry and heldout checks still occur before a consumed draw.',
            'Full native/durable storage lifecycle and actual process closure require separate validation.'],
        'actual_audio_native_GPU_or_storage_lifecycle_proved': False,
        'production_enabled': False, 'launch_ready': False}


def from_pinned_metadata(*, inputs, limits, disk):
    require(set(inputs) == {'population', 'geometry', 'cleaner', 'replay'}, 'Four exact metadata authorities')
    values = {name: small_metadata(pin) for name, pin in inputs.items()}
    result = derive(**values, limits=limits, disk=disk)
    result['metadata_inputs'] = deepcopy(inputs)
    return result
