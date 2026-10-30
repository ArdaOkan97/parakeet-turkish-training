"""NEW native-readable recorded-room benchmark: TWO real aggregate leaves.

Original bridge7 planning and sink7 numerical/file engines remain frozen.
This explicit benchmark namespace cannot adopt the existing shared
GPU oracle. Its ordinary reader API is unchanged; native admission is separate.
"""
from copy import deepcopy
from pathlib import Path
import re
import subprocess

import fastmss_phase_aware_feature_window_v7 as engine
import fastmss_owned_process_closure_v2 as process
import fastmss_recorded_room_train_v1 as augmentation

bridge = engine.bridge
ROOT, PROJECT = engine.ROOT, engine.PROJECT
CACHE = ROOT / 'temp/ASR18_recorded_room_CPU_benchmark_v1'
CAPACITY = engine.CAPACITY
CONTRACT = 'FastMSS_recorded_room_CPU_source_and_numerical_contract_v1'
REQUEST = 'FastMSS_recorded_room_CPU_preparation_request_v1'
MODE = 'recorded_room_first4_native_benchmark_only'
OWNER = 'FastMSS_owned_recorded_room_CPU_feature_window_v1'
REFERENCE = 'FastMSS_recorded_room_independent_CPU_reference_window_v1'
REFERENCE_MEMBER = 'FastMSS_recorded_room_reference_update_within_actual_CPU_window_v1'
FRONTEND = 'FastMSS_recorded_room_full_CPU_frontend_window_v1'
UPDATE = 'FastMSS_recorded_room_update_within_actual_frontend_window_v1'
CLOSURE = 'FastMSS_actual_recorded_room_CPU_frontend_window_closure_v1'
MANIFEST = 'FastMSS_actual_recorded_room_closed_CPU_feature_window_v1'
canonical, digest, same, require = engine.canonical, engine.digest, engine.same, engine.require
record, read_bound, write_new, scoped, serial, integer = engine.record, engine.read_bound, engine.write_new, engine.scoped, engine.serial, engine.integer


def capacity_lock():
    return engine._lock(CACHE.with_suffix('.capacity.lock'))


def verify_source_contract(pin, *, guard):
    value = read_bound(pin, guard)
    require(value['kind'] == CONTRACT and value['mode'] == MODE and value['production_enabled'] is False
        and value['production_source_eligible'] is False and value['validation_only'] is False
        and value['native_benchmark_only'] is True,
        'Exact production-disabled native-benchmark source contract')
    paths = {'protocol': Path(__file__), 'engine': Path(engine.__file__),
        'reference': PROJECT / 'work/research/produce_FastMSS_recorded_room_reference_CPU_v1.py',
        'frontend': PROJECT / 'work/research/produce_FastMSS_recorded_room_frontend_CPU_v1.py',
        'controller': PROJECT / 'work/research/run_ASR18_recorded_room_CPU_window_v1.py',
        'core': PROJECT / 'outputs/parakeet_turkish/fastmss_row_serial_cpu_v8.py',
        'guard': PROJECT / 'work/research/ASR18_bounded_CPU_phase_v3.py',
        'process': Path(process.__file__), 'augmentation_adapter': Path(augmentation.__file__),
        'real_noise_mixer': Path(augmentation.mixer.__file__), 'real_noise_decoder': Path(augmentation.decoder.__file__),
        'room_provenance': Path(augmentation.rooms.__file__)}
    require(set(value['sources']) == set(paths), 'Exact finite new source table')
    for key, path in paths.items():
        require(same(record(path, guard), value['sources'][key]), 'New aggregate CPU source differs: ' + key)
    require(type(value['maximum_window_updates']) is int and value['maximum_window_updates'] == 20
        and value['feature_capacity_bytes'] == CAPACITY and value['raw_PCM_capacity_bytes'] == 128 * 1024**2
        and type(value['raw_PCM_maximum_jobs']) is int and value['raw_PCM_maximum_jobs'] == 2
        and value['one_actual_reference_window0_before_frontend'] is True and value['per_update_child0_claimed'] is False
        and type(value['freeze_after_setup']) is bool and value['process_owner'] == '/root/fastmss_integration',
        'Unchanged finite feature/raw caps and truthful aggregate closure scope')
    review = value['resource_review']
    require(same(review, {'leaf_maximum_private_bytes': 1_800_000_000, 'controller_private_reserve_bytes': 64 * 1024**2,
        'inner_shim_private_reserve_bytes': 32 * 1024**2, 'outer_shim_private_reserve_bytes': 32 * 1024**2,
        'whole_inner_family_bytes': 1_900_663_296, 'whole_family_forecast_bytes': 1_934_217_728}),
        'No resource-margin/CPU architecture changes in this source contract')
    plan = engine._load_bridge(value['bridge'], guard)
    require(same(read_bound(plan['session'], guard)['CPU_preparation_contract'], value['numerical_CPU8_contract']),
        'Old numerical contract is immutable input, never relabeled new launcher authority')
    _, identity = augmentation.policy_identity(value['augmentation_policy'], guard)
    require(same(identity, value['augmentation_identity']) and value['original_augmentation_identity'] == plan['augmentation_identity']
        and plan['phase'] == bridge.WEAK and len(plan['updates']) == 4
        and sum(len(v['rows']) for v in plan['updates']) == 64,
        'Explicit new augmentation overlay; original schedule/session/rows remain immutable')
    return value, plan


def _check_reference_rows(update, references, guard):
    """Same original bridge7 reference equations; aggregate closure is separate."""
    require(type(references) is list and len(references) == len(update['rows']), 'All ordered independent references')
    for entry, reference in zip(update['rows'], references):
        require(reference['original_row_sha256'] == entry['original_row_sha256']
            and same(reference['context'], entry['context']) and same(reference['full_target_ids'], entry['full_target_ids'])
            and reference['target_ids_u32_sha256'] == entry['target_ids_u32_sha256'], 'Exact independent full target/source/context')
        audio = reference['expected_audio']
        require(type(audio) is dict and set(audio) == {'sample_rate', 'samples', 'waveform_sha256'}
            and type(audio['sample_rate']) is int and audio['sample_rate'] == 16000 and bridge.sha(audio['waveform_sha256']),
            'Whole original float32 mono16k PCM identity')
        integer(audio['samples'], 1)
        require(reference['whole_original_clip'] is True and reference['augmentation_applied'] is False
            and reference['actual_geometry_and_heldout_checked'] is True, 'Actual whole original geometry/heldout proof')
        sources = reference['original_source_files']
        require(type(sources) is list and 1 <= len(sources) <= 4, 'Complete independent source inventory')
        for source in sources:
            require(type(source) is dict and set(source) in ({'path', 'bytes', 'sha256'}, {'path', 'bytes', 'sha256', 'mtime_ns'})
                and bridge.sha(source['sha256']), 'Exact source identity pin')
            integer(source['bytes'], 1); path = scoped(source['path'], source=True)
            require(path.stat().st_size == source['bytes'] and ('mtime_ns' not in source
                or type(source['mtime_ns']) is int and path.stat().st_mtime_ns == source['mtime_ns']), 'Original source signature changed')
        if entry['role'] == 'replay':
            cache = reference['genuine_teacher_feature_cache']
            require(type(cache) is dict and cache['source_id'] == entry['source_id']
                and cache['target_ids_u32_sha256'] == entry['target_ids_u32_sha256']
                and cache['clean_waveform_sha256'] == audio['waveform_sha256'] and bridge.sha(cache['genuine_feature_sha256'])
                and reference['teacher_or_replay_waveform_changed'] is False, 'Original genuine replay features stay unchanged')


def verify_reference_window(report_pin, actual0, *, contract_pin, guard):
    contract, plan = verify_source_contract(contract_pin, guard=guard)
    value = read_bound(report_pin, guard)
    process.verify(read_bound(actual0, guard), kind='ASR18_owned_CPU_process_completion_v1',
        owner=contract['process_owner'], report=report_pin, guard=guard)
    require(value['kind'] == REFERENCE and value['passed'] is True and value['mode'] == MODE
        and same(value['bridge'], contract['bridge']) and same(value['source_contract'], contract_pin)
        and same(value['worker_source'], contract['sources']['reference'])
        and same(value['parent_source'], contract['sources']['controller'])
        and value['all_updates_reported'] is True and type(value['actual_optimizer_updates']) is int
        and value['actual_optimizer_updates'] == 0 and value['sampler_journal_frontier_advanced'] is False
        and value['per_update_child0_claimed'] is False and value['raw_lifetime_receipt']['passed'] is True,
        'Genuine complete independent aggregate reference0, exact new worker/parent/source contract')
    require(type(value['updates']) is list and len(value['updates']) == len(plan['updates']), 'No missing/extra aggregate reference member')
    reports = []
    for index, member in enumerate(value['updates']):
        require(type(member['update_index']) is int and member['update_index'] == index, 'Exact reference table order')
        item = read_bound(member['report'], guard)
        require(item['kind'] == REFERENCE_MEMBER and item['passed'] is True and type(item['update_index']) is int
            and item['update_index'] == index and same(item['producer'], value['producer'])
            and same(item['worker_source'], value['worker_source']) and same(item['parent_source'], value['parent_source'])
            and same(item['request'], value['request']) and type(item['actual_optimizer_updates']) is int
            and item['actual_optimizer_updates'] == 0 and item['sampler_journal_frontier_advanced'] is False
            and same(item['source_contract'], contract_pin) and same(item['bridge'], contract['bridge'])
            and item['phase'] == plan['phase'] and item['per_update_child0_claimed'] is False,
            'Selected independent update is a member of the actual same-process aggregate0')
        _check_reference_rows(plan['updates'][index], item['references'], guard)
        reports.append(item)
    return value, reports


def reference_member_binding(report_pin, actual0, index, *, contract_pin, guard):
    value, _ = verify_reference_window(report_pin, actual0, contract_pin=contract_pin, guard=guard)
    integer(index, 0, len(value['updates']) - 1)
    return {'kind': 'FastMSS_selected_reference_member_of_actual_window0_v1',
        'aggregate_report': report_pin, 'aggregate_actual0': actual0, 'update_index': index,
        'member_report': value['updates'][index]['report'], 'source_contract': contract_pin,
        'per_update_child0_claimed': False}


def _reference_member(binding, owner, index, guard):
    require(binding['kind'] == 'FastMSS_selected_reference_member_of_actual_window0_v1'
        and type(binding['update_index']) is int and binding['update_index'] == index
        and same(binding['source_contract'], owner['source_contract']) and binding['per_update_child0_claimed'] is False,
        'Exact aggregate-selected reference member, no per-update process proof')
    aggregate, reports = verify_reference_window(binding['aggregate_report'], binding['aggregate_actual0'],
        contract_pin=owner['source_contract'], guard=guard)
    require(same(binding['member_report'], aggregate['updates'][index]['report']), 'Reference member substituted')
    return reports[index]


def create_window(contract_pin, *, window_id, frontend, guard, verify_prior_boundary=None):
    serial(); contract, plan = verify_source_contract(contract_pin, guard=guard)
    original = engine._load_bridge(contract['bridge'], guard)
    rebuilt = bridge.bridge_plan(original['coordinator_plan'], DATA_evidence=original['DATA_evidence'],
        session_binding=original['session'], window_index=original['window_index'],
        prior_window_boundary=original['prior_window_boundary'], verify_prior_boundary=verify_prior_boundary, guard=guard)
    require(same(rebuilt, plan), 'Full original DATA/session/draw/phase/tail plan must independently revalidate')
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', window_id) is not None, 'Fresh exact safe window ID')
    engine._frontend(frontend, guard)
    names = ['owner.json', 'manifest.json']
    for i, update in enumerate(plan['updates']):
        names += [f'active_{i:02}.json', f'update_{i:02}.json']
        for j in range(len(update['rows'])): names += [f'u{i:02}_r{j:02}_features.npy', f'u{i:02}_r{j:02}_mask.npy']
    owner = {'kind': OWNER, 'source': record(__file__, guard), 'file_engine_source': record(engine.__file__, guard),
        'source_contract': contract_pin, 'window_id': window_id, 'bridge': contract['bridge'], 'session': plan['session'],
        'window_index': plan['window_index'], 'phase': plan['phase'], 'updates': len(plan['updates']), 'owned_names': names,
        'capacity_bytes': CAPACITY, 'augmentation_identity': contract['augmentation_identity'], 'frontend': frontend,
        'original_augmentation_identity': plan['augmentation_identity'], 'native_benchmark_only': True,
        'raw_PCM_capacity_bytes': 128 * 1024**2, 'raw_PCM_maximum_jobs': 2, 'prepared_is_committed': False,
        'sampler_journal_frontier_advanced': False, 'production_enabled': False, 'production_source_eligible': False,
        'namespace_role': MODE, 'two_actual_aggregate_leaves_required': True, 'per_update_child0_claimed': False}
    with capacity_lock():
        CACHE.mkdir(parents=True, exist_ok=True)
        require(not list(CACHE.iterdir()), 'Preserve any outstanding NEW recorded-room window; no old namespace adoption')
        folder = CACHE / window_id; folder.mkdir(exist_ok=False); write_new(folder / 'owner.json', owner)
    return record(folder / 'owner.json', guard)


def owner_folder(pin, guard):
    owner = read_bound(pin, guard); folder = scoped(pin['path']).parent
    require(owner['kind'] == OWNER and same(owner['source'], record(__file__, guard))
        and same(owner['file_engine_source'], record(engine.__file__, guard)) and folder.parent == CACHE.resolve()
        and folder.name == owner['window_id'] and Path(pin['path']).name == 'owner.json'
        and owner['production_enabled'] is False and owner['production_source_eligible'] is False
        and owner['namespace_role'] == MODE and owner['capacity_bytes'] == CAPACITY
        and owner['raw_PCM_capacity_bytes'] == 128 * 1024**2 and owner['raw_PCM_maximum_jobs'] == 2,
        'Exact fresh native-benchmark aggregate owner/source/namespace/caps')
    _, plan = verify_source_contract(owner['source_contract'], guard=guard)
    require(same(owner['bridge'], read_bound(owner['source_contract'], guard)['bridge'])
        and same(owner['session'], plan['session']) and owner['phase'] == plan['phase']
        and owner['window_index'] == plan['window_index'] and owner['updates'] == len(plan['updates'])
        and same(owner['original_augmentation_identity'], plan['augmentation_identity'])
        and same(owner['augmentation_identity'], read_bound(owner['source_contract'], guard)['augmentation_identity'])
        and owner['native_benchmark_only'] is True, 'Owner/original bridge and explicit new augmentation overlay differ')
    return owner, folder, plan


class WindowProducer:
    """One live frontend prepares all updates serially; aggregate exit closes them."""
    def __init__(self, owner_pin, *, reference_report, reference_actual0, clean_reader,
                 weak_transform, transform_identity, frontend, guard):
        serial(); self.owner_pin = owner_pin; self.guard = guard
        self.owner, folder, self.plan = owner_folder(owner_pin, guard)
        self.reference_report, self.reference_actual0 = reference_report, reference_actual0
        aggregate, self.references = verify_reference_window(reference_report, reference_actual0,
            contract_pin=self.owner['source_contract'], guard=guard)
        self.reference_table = aggregate['updates']; self.producer = process.identity()
        self.store = engine._Store(folder, self.owner, guard, capacity_lock)
        self.protocols = [engine._RowProtocol(update, refs['references'], self.owner['augmentation_identity'], self.store,
            clean_reader=clean_reader, weak_transform=weak_transform, transform_identity=transform_identity,
            frontend=frontend, guard=guard) for update, refs in zip(self.plan['updates'], self.references)]
        self.cursor = 0; self.active = None; self.parts = []

    def begin_update(self, index):
        serial(); require(type(index) is int and index == self.cursor and self.active is None
            and same(process.identity(), self.producer), 'Same sole actual frontend, exact next update')
        active = {'kind': 'FastMSS_update_within_live_frontend_window_v1', 'owner': self.owner_pin,
            'producer': self.producer, 'update_index': index, 'prepared_is_committed': False,
            'per_update_child0_claimed': False, 'actual_aggregate_exit_required': True}
        self.active = self.store.write_metadata(f'active_{index:02}.json', active)
        return self.protocols[index]

    def finish_update(self, index):
        serial(); require(type(index) is int and index == self.cursor and self.active is not None
            and same(process.identity(), self.producer), 'Same actual ordered frontend update')
        rows, payload = self.protocols[index].prepared(); update = self.plan['updates'][index]; original = update['coordinator_update']
        binding = {'kind': 'FastMSS_selected_reference_member_of_actual_window0_v1',
            'aggregate_report': self.reference_report, 'aggregate_actual0': self.reference_actual0,
            'update_index': index, 'member_report': self.reference_table[index]['report'],
            'source_contract': self.owner['source_contract'], 'per_update_child0_claimed': False}
        item = {'kind': UPDATE, 'owner': self.owner_pin, 'active_record': self.active, 'producer': self.producer,
            'update_index': index, 'global_update': original['global_update'], 'phase': original['phase'],
            'batch_id': original['batch_id'], 'native_transaction_sha256': original['native_transaction_sha256'],
            'before_sampler_sha256': original['before_sampler_sha256'], 'after_sampler_sha256': original['after_sampler_sha256'],
            'coordinator_update_sha256': update['coordinator_update_sha256'], 'rows': rows, 'prepared_payload': payload,
            'reference_window_member': binding, 'frontend': self.owner['frontend'],
            'augmentation_identity': self.owner['augmentation_identity'], 'prepared_is_committed': False,
            'sampler_journal_frontier_advanced': False, 'actual_CPU_aggregate_exit_observed': False,
            'per_update_child0_claimed': False, 'raw_processing': 'serial_one_row_within_finite_window'}
        pin = self.store.write_metadata(f'update_{index:02}.json', item)
        self.parts.append({'update_index': index, 'active_record': self.active, 'update_record': pin})
        self.active = None; self.cursor += 1
        return pin

    def prepared(self):
        require(self.active is None and self.cursor == len(self.plan['updates'])
            and same(process.identity(), self.producer), 'Every full ordered update prepared by actual same frontend')
        return deepcopy(self.parts)


def verify_feature_window(pin, owner_pin, *, guard):
    value = read_bound(pin, guard); owner, _, plan = owner_folder(owner_pin, guard)
    contract = read_bound(owner['source_contract'], guard)
    require(value['kind'] == CLOSURE and same(value['owner'], owner_pin)
        and same(value['observer_source'], record(__file__, guard)) and value['per_update_child0_claimed'] is False,
        'New actual aggregate frontend closure only, never old per-update kind')
    actual = process.verify(read_bound(value['actual_process0'], guard), kind='ASR18_owned_CPU_process_completion_v1',
        owner=contract['process_owner'], report=value['report'], guard=guard)
    report = read_bound(value['report'], guard)
    require(report['kind'] == FRONTEND and report['mode'] == MODE and report['passed'] is True and same(report['owner'], owner_pin)
        and same(report['source_contract'], owner['source_contract']) and same(report['producer'], actual['producer'])
        and same(report['worker_source'], contract['sources']['frontend'])
        and same(report['parent_source'], contract['sources']['controller']) and report['all_updates_reported'] is True
        and type(report['actual_optimizer_updates']) is int and report['actual_optimizer_updates'] == 0
        and report['sampler_journal_frontier_advanced'] is False and report['raw_lifetime_receipt']['passed'] is True
        and report['per_update_child0_claimed'] is False and report['freeze_after_setup'] == contract['freeze_after_setup']
        and type(report['full_rows']) is int and report['full_rows'] == sum(len(v['rows']) for v in plan['updates'])
        and type(report['new_NPY_files_written']) is int and report['new_NPY_files_written'] == 2 * report['full_rows']
        and report['native_features_ready'] is False and same(value['updates'], report['updates'])
        and len(report['updates']) == len(plan['updates']), 'Complete actual frontend window source/report/table')
    for i, part in enumerate(report['updates']):
        require(type(part['update_index']) is int and part['update_index'] == i, 'No missing/duplicate/reordered frontend table member')
        active, item = read_bound(part['active_record'], guard), read_bound(part['update_record'], guard)
        require(active['kind'] == 'FastMSS_update_within_live_frontend_window_v1'
            and same(active['owner'], owner_pin) and same(active['producer'], actual['producer'])
            and type(active['update_index']) is int and active['update_index'] == i
            and active['per_update_child0_claimed'] is False and same(item['producer'], actual['producer'])
            and same(item['active_record'], part['active_record']) and type(item['update_index']) is int
            and item['update_index'] == i and item['per_update_child0_claimed'] is False,
            'All live per-update records belong to ONE actually closed frontend producer')
    return value


def observe_feature_window(child, *, owner_pin, actual_process0, report, output, guard):
    serial(); require(isinstance(child, subprocess.Popen), 'Actual owned aggregate frontend Popen required')
    code = child.wait(timeout=5); require(type(code) is int and code == 0, 'Actual aggregate frontend handle wait0')
    actual = read_bound(actual_process0, guard)
    require(actual['launcher']['pid'] == child.pid and same(actual['report'], report), 'Same actual handle/report observer chain')
    value = {'kind': CLOSURE, 'owner': owner_pin, 'actual_process0': actual_process0, 'report': report,
        'updates': read_bound(report, guard)['updates'], 'observer_source': record(__file__, guard),
        'per_update_child0_claimed': False}
    output = scoped(output); require(not output.is_relative_to(CACHE.resolve()), 'Actual closure outside future transient slot')
    write_new(output, value); pin = record(output, guard)
    verify_feature_window(pin, owner_pin, guard=guard)
    return pin


def _validate_update(owner_pin, item, update, store):
    require(item['kind'] == UPDATE and same(item['owner'], owner_pin) and item['phase'] == update['phase']
        and type(item['update_index']) is int and item['update_index'] == update['update_index']
        and item['coordinator_update_sha256'] == update['coordinator_update_sha256']
        and item['prepared_is_committed'] is False and item['sampler_journal_frontier_advanced'] is False
        and item['per_update_child0_claimed'] is False, 'Exact new aggregate-compatible full update')
    original = update['coordinator_update']
    require(same(item['frontend'], store.owner['frontend']) and same(item['augmentation_identity'], store.owner['augmentation_identity'])
        and all(same(item[k], original[k]) for k in ('global_update', 'batch_id', 'native_transaction_sha256', 'before_sampler_sha256', 'after_sampler_sha256'))
        and len(item['rows']) == len(update['rows']), 'Full original frontend/source/transaction identity')
    references = _reference_member(item['reference_window_member'], store.owner, update['update_index'], store.guard)['references']
    payload = item['prepared_payload']
    require(payload['state'] == 'prepared_not_committed' and payload['committed_frontier_advanced'] is False
        and digest(payload['rows']) == payload['payload_sha256'] and len(payload['rows']) == len(item['rows']), 'Full copied uncommitted payload')
    for entry, expected, event, ref in zip(item['rows'], update['rows'], payload['rows'], references):
        require(all(same(entry[k], expected[k]) for k in ('source_id', 'language', 'context', 'role', 'full_target_ids',
            'target_ids_u32_sha256', 'raw_text_sha256', 'original_row_sha256')) and same(event['draw'], expected['context']),
            'Complete immutable ordered target/source/context')
        require(event['augmentation']['policy_sha256'] == store.owner['augmentation_identity']['policy_sha256']
            and entry['clean_PCM_sha256'] == ref['expected_audio']['waveform_sha256'] and type(entry['samples']) is int
            and entry['samples'] == ref['expected_audio']['samples'] and type(entry['sample_rate']) is int
            and entry['sample_rate'] == 16000 and bridge.sha(entry['prepared_PCM_sha256']), 'Whole clean/prepared PCM identity')
        protected = update['phase'] == bridge.CLEANER or expected['role'] == 'replay' or entry['samples'] > 120 * 16000
        require(not protected or event['augmentation']['applied'] is False
            and entry['clean_PCM_sha256'] == entry['prepared_PCM_sha256'], 'Cleaner/replay/long whole clips remain clean')
        store.validate_arrays(entry)
    require(update['phase'] != bridge.CLEANER or payload['protection_kind'] == engine.CLEAN_PROTECTION, 'Exact cleaner protection')


def finalize_window(owner_pin, closure_pin, *, guard):
    serial(); owner, folder, plan = owner_folder(owner_pin, guard)
    closed = verify_feature_window(closure_pin, owner_pin, guard=guard)
    store = engine._Store(folder, owner, guard, capacity_lock); parts = []
    for i, part in enumerate(closed['updates']):
        _validate_update(owner_pin, read_bound(part['update_record'], guard), plan['updates'][i], store)
        parts.append({'update': part['update_record'], 'aggregate_member_index': i})
    manifest = {'kind': MANIFEST, 'owner': owner_pin, 'bridge': owner['bridge'], 'session': owner['session'],
        'phase': owner['phase'], 'updates': parts, 'actual_aggregate_frontend_closure': closure_pin,
        'capacity_bytes': CAPACITY, 'prepared_is_committed': False, 'sampler_journal_frontier_advanced': False,
        'augmentation_identity': owner['augmentation_identity'], 'frontend': owner['frontend'],
        'per_update_child0_claimed': False, 'production_enabled': False, 'production_source_eligible': False,
        'validation_only_array_verifier': {'source': record(engine.stdlib_arrays.__file__, guard),
            'NumPy_required': False, 'maximum_payload_read_bytes': engine.stdlib_arrays.CHUNK_BYTES,
            'native_take_row_numpy_behavior_changed': False}}
    with capacity_lock():
        used = store.inventory(); manifest['aggregate_files_bytes'] = used
        manifest['aggregate_file_count'] = len(list(folder.iterdir())) + 1
        for _ in range(10):
            total = used + len(canonical(manifest))
            if total == manifest['aggregate_files_bytes']: break
            manifest['aggregate_files_bytes'] = total
        require(used + len(canonical(manifest)) == manifest['aggregate_files_bytes'] <= CAPACITY, 'Actual finalized whole-window cap')
        write_new(folder / 'manifest.json', manifest)
    return record(folder / 'manifest.json', guard)


class FeatureWindowReader(engine._ReadCursor):
    """Unchanged consume-once actual NumPy reads; NEW aggregate proof binding."""
    def __init__(self, owner_pin, manifest_pin, *, guard, resume_binding=None, verify_checkpoint=None):
        serial(); self.owner_pin = owner_pin; self.owner, self.folder, self.plan = owner_folder(owner_pin, guard)
        self.manifest = read_bound(manifest_pin, guard)
        require(self.manifest['kind'] == MANIFEST and same(self.manifest['owner'], owner_pin)
            and same(self.manifest['bridge'], self.owner['bridge']) and same(self.manifest['session'], self.owner['session'])
            and self.manifest['phase'] == self.owner['phase'] and self.manifest['prepared_is_committed'] is False
            and len(self.manifest['updates']) == self.owner['updates'] and self.manifest['per_update_child0_claimed'] is False,
            'Complete actual aggregate manifest, never an old per-child manifest alias')
        self.closed = verify_feature_window(self.manifest['actual_aggregate_frontend_closure'], owner_pin, guard=guard)
        count = 0
        if resume_binding is not None:
            count = bridge.validate_resume(self.owner['bridge'], resume_binding, verify_checkpoint=verify_checkpoint, guard=guard)['next_update_index']
        super().__init__(engine._Store(self.folder, self.owner, guard, capacity_lock), self._verified_update,
                         updates=self.owner['updates'], guard=guard, start=count)

    def _verified_update(self, index):
        integer(index, 0, self.owner['updates'] - 1)
        require(same(read_bound(self.manifest['actual_aggregate_frontend_closure'], self.guard), self.closed),
            'Pinned actual aggregate closure changed after reader admission')
        report = read_bound(self.closed['report'], self.guard)
        require(same(report['updates'], self.closed['updates']), 'Actual aggregate report table changed after admission')
        part, actual = self.manifest['updates'][index], self.closed['updates'][index]
        require(type(part['aggregate_member_index']) is int and part['aggregate_member_index'] == index
            and same(part['update'], actual['update_record']), 'Selected native update is in the actual aggregate0')
        return deepcopy(part['update'])


def cleanup_window(owner_pin, *, closure_pin, manifest_pin, boundary, verify_boundary, evidence_path, guard):
    """Exact NEW owner only; original/current shared namespace is unreachable."""
    serial(); owner, folder, plan = owner_folder(owner_pin, guard)
    verify_feature_window(closure_pin, owner_pin, guard=guard)
    require(boundary['kind'] in ('coherent_checkpoint_boundary', 'discard_uncommitted_feature_window')
        and boundary['owner_sha256'] == owner_pin['sha256'] and same(boundary['session'], owner['session'])
        and boundary['phase'] == owner['phase'] and callable(verify_boundary), 'Exact actual caller-owned boundary')
    if boundary['kind'] == 'coherent_checkpoint_boundary':
        require(manifest_pin is not None and type(boundary['committed_window_updates']) is int
            and boundary['committed_window_updates'] == owner['updates']
            and boundary['window_manifest_sha256'] == manifest_pin['sha256'], 'No intermediate future arrays retirement')
        pointer = read_bound(boundary['coherent_checkpoint_pointer'], guard)
        require(pointer['all_window_updates_committed'] is True and pointer['committed_window_updates'] == owner['updates']
            and pointer['identity']['window']['plan_sha256'] == plan['coordinator_plan_sha256']
            and pointer['saved_semantic_state_verified']['passed'] is True and same(pointer['checkpoint'], boundary['checkpoint'])
            and same(read_bound(boundary['checkpoint'], guard)['payload'], pointer['payload']), 'Exact coherent all-state/payload/window')
        require(('actual_native_closed' in boundary) != ('resident_native_boundary' in boundary), 'One actual exit or distinct actual live idle')
        if 'resident_native_boundary' in boundary:
            process.verify_resident_idle(read_bound(boundary['resident_native_boundary'], guard), pointer=pointer,
                proof_binding=boundary['resident_native_boundary'], manifest=manifest_pin, phase=owner['phase'], guard=guard)
        else: bridge.observed_native0(read_bound(boundary['actual_native_closed'], guard), guard)
    else:
        require(type(boundary['native_work_started']) is bool, 'Truthful never-launched or actually closed discard')
        if boundary['native_work_started']:
            native = read_bound(boundary['actual_native_closed'], guard)
            require(native['owner'] == '/root/memory_diagnosis' and type(native['actual_native_exit_code']) is int
                and type(native['actual_tool_exit_code']) is int and type(native['actual_tool_chunk_id']) is str
                and native['actual_tool_chunk_id'], 'Actual failed native closed proof, never inferred from idle')
    require(verify_boundary(deepcopy(boundary)) is True, 'Actual source-bound native owner callback required')
    if manifest_pin is not None:
        reader = FeatureWindowReader(owner_pin, manifest_pin, guard=guard)
        for index, update in enumerate(plan['updates']):
            _validate_update(owner_pin, read_bound(reader.update_record(index), guard), update, reader.store)
    else: require(boundary['kind'] == 'discard_uncommitted_feature_window', 'Incomplete window only truthful discard')
    evidence_path = scoped(evidence_path)
    require(not evidence_path.is_relative_to(CACHE.resolve()), 'Durable archive outside new transient slot')
    store = engine._Store(folder, owner, guard, capacity_lock)
    removed, archive = engine._retire_files(store, archive={'kind': 'FastMSS_aggregate_CPU_window_durable_metadata_v1',
        'owner_binding': owner_pin, 'manifest_binding': manifest_pin, 'boundary': boundary,
        'actual_aggregate_closure': closure_pin, 'per_update_child0_claimed': False}, evidence_path=evidence_path)
    return {'kind': 'FastMSS_aggregate_CPU_window_owned_retirement_v1', 'passed': True, 'owner': owner_pin,
        'removed_exact_owned_files': removed, 'preserved_metadata': archive, 'boundary_sha256': digest(boundary),
        'original_shared_oracle_untouched': True, 'sampler_journal_frontier_advanced': False}
