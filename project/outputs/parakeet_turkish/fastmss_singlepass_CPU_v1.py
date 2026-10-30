"""Bounded single-decode feature windows. Preparation never commits a draw.

The CPU producer reports the checks it actually performs, rather than inventing
an independent reference process. A parent seals the manifest only after a real
owned process exit. A provisional successor additionally needs actual adoption.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

import fastmss_BPE9223_feature_engine_v12 as engine
import fastmss_BPE9223_cpu_common_v12 as core
import fastmss_BPE9223_plan_bridge_v12 as bridge
from fastmss_owned_process_closure_v2 import identity, verify

ROOT = Path(r'E:\gpu_training_scratchpad')
HERE = Path(__file__).resolve().parent
REQUEST = 'ASR18_BPE9223_singlepass_CPU_request_v1'
OWNER = 'FastMSS_single_decode_feature_owner_v1'
REPORT = 'FastMSS_single_decode_prepared_window_report_v1'
MANIFEST = 'FastMSS_single_decode_actual_closed_feature_window_v1'
UPDATE = 'FastMSS_single_decode_provisional_update_v1'
CAPACITY = 512 * 1024**2
RAW_CAPACITY = 128 * 1024**2
canonical, digest, same, require = bridge.canonical, bridge.digest, bridge.same, bridge.require
record, read_bound, write_new = engine.record, engine.read_bound, engine.write_new


def exact_slot(request):
    run = Path(request['run_namespace']).resolve()
    require(run.is_relative_to((ROOT / 'runs').resolve()), 'Existing reviewed E run only')
    index = request['slot_index']
    require(type(index) is int and index in (0, 1), 'Only two bounded feature slots')
    slot = run / 'provisional' / ('slot_' + str(index))
    require(Path(request['output_root']).resolve() == slot, 'Exact stable owned slot, no future AUX reservation')
    for path in (slot, *slot.parents):
        if path.exists():
            info = path.lstat()
            require(not path.is_symlink() and not info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                    'No reparse or symbolic ownership escape')
        if path == ROOT: break
    return slot


def validate_request(pin, guard):
    request = read_bound(pin, guard)
    require(request['kind'] == REQUEST and request['production_enabled'] is False
            and request['prepared_is_committed'] is False, 'Typed non-committing CPU request')
    exact_slot(request)
    review = read_bound(request['ROOT_review'], guard)
    require(review['accepted'] is True and review['reviewer'] == '/root'
            and review['scope'] == 'BPE9223_single_decode_bounded_CPU_v1'
            and review['worker_source'] == request['worker_source']
            and review['protocol_source'] == request['protocol_source'] == record(__file__, guard)
            and review['numerical_CPU_contract'] == request['numerical_CPU_contract']
            and review['run_namespace'] == request['run_namespace'], 'Actual source/numerical/root CPU authority')
    session = read_bound(request['session'], guard)
    activation = read_bound(request['augmentation_ROOT_review'], guard)
    require(activation['accepted'] is True and activation['reviewer'] == '/root'
            and activation['scope'] == 'BPE9223_recorded_noise_16room_bounded_CPU_window_v1'
            and activation['configuration'] == request['augmentation_configuration']
            and review['augmentation_ROOT_review'] == request['augmentation_ROOT_review']
            and review['augmentation_configuration'] == request['augmentation_configuration'],
            'Preserve genuine original recorded-room activation and separately reviewed new transport')
    contract = core.verify_contract(request['numerical_CPU_contract'], session, guard=guard)
    validated = bridge.validate_DATA(request['DATA_evidence'], guard)
    plan = read_bound(request['plan'], guard)
    normalized = bridge.normalize_updates(plan, population=validated['retained_rows'], guard=guard)
    require(1 <= len(normalized) <= 20 and plan['phase'] in (bridge.WEAK, bridge.CLEANER)
            and request['window_index'] >= 0 and request['augmentation_identity'] == review['augmentation_identity']
            and request['original_augmentation_identity'] == session['augmentation_identity'],
            'Full bounded actual plan and unchanged augmentation')
    predecessor = request.get('provisional_predecessor')
    if predecessor is not None:
        require(predecessor['kind'] == 'ASR18_private_next_window_prediction_v1'
                and predecessor['committed_frontier_advanced'] is False
                and predecessor['expected_optimizer_updates'] == plan['initial_committed_updates']
                and predecessor['expected_sampler_sha256'] == plan['initial_sampler_sha256']
                and type(predecessor['command_nonce']) is str and len(predecessor['command_nonce']) >= 16,
                'Private predecessor prediction, never a fabricated future checkpoint')
        prior = read_bound(predecessor['current_plan'], guard)
        require(digest(prior['final_private_planning_sampler']) == plan['initial_sampler_sha256'],
                'Successor starts at the actual predecessor private final sampler')
    else:
        actual = read_bound(request['current_checkpoint_pointer'], guard)
        require(actual['committed_updates'] == plan['initial_committed_updates']
                and actual['sampler_state_sha256'] == plan['initial_sampler_sha256'],
                'Initial window starts at its real recovered checkpoint')
    return request, session, contract, validated, plan, normalized


class TransientStore(engine._Store):
    """Same NPY bytes; transient caches need no per-array device fsync barrier."""
    def save(self, name, array):
        import numpy as np
        require(name in self.owner['owned_names'] and re.fullmatch(r'u\d{2}_r\d{2}_(features|mask)\.npy', name),
                'Explicit owned array name')
        require(isinstance(array, np.ndarray) and array.flags.c_contiguous
                and array.dtype in (np.float32, np.bool_) and not array.dtype.hasobject,
                'Unchanged finite numeric NPY representation')
        with self.lock():
            require(self.inventory() + array.nbytes + 1024 <= CAPACITY, 'Unchanged whole-window capacity before allocation')
            with (self.folder / name).open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
            require(self.inventory() <= CAPACITY, 'Actual closed transient NPY capacity')
        return {'file': record(self.folder / name, self.guard), 'shape': list(array.shape),
                'dtype': str(array.dtype), 'array_sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def create_window(request_pin, frontend, guard, *, validated_inputs=None):
    request, session, contract, validated, plan, normalized = (validate_request(request_pin, guard)
        if validated_inputs is None else validated_inputs)
    folder = exact_slot(request)
    names = ['owner.json']
    for i, update in enumerate(normalized):
        names.append('update_%02d.json' % i)
        for j in range(len(update['rows'])):
            names += ['u%02d_r%02d_features.npy' % (i, j), 'u%02d_r%02d_mask.npy' % (i, j)]
    engine._frontend(frontend, guard)
    require(len(names) <= 704, 'Bounded exact owned inventory')
    owner = {'kind': OWNER, 'source': record(__file__, guard), 'request': request_pin,
             'plan': request['plan'], 'session': request['session'], 'window_id': request['window_id'],
             'slot_index': request['slot_index'], 'phase': plan['phase'], 'updates': len(normalized),
             'capacity_bytes': CAPACITY, 'raw_PCM_capacity_bytes': RAW_CAPACITY,
             'raw_PCM_maximum_jobs': 2, 'frontend': frontend, 'owned_names': names,
             'augmentation_identity': request['augmentation_identity'], 'prepared_is_committed': False,
             'sampler_journal_frontier_advanced': False, 'one_actual_combined_CPU_exit_required': True}
    folder.parent.mkdir(parents=True, exist_ok=True)
    with engine._lock(folder.parent / 'slots.lock'):
        require(not folder.exists(), 'Preserve an occupied slot, no implicit adoption or deletion')
        occupied = [p for p in folder.parent.iterdir() if p.is_dir()]
        require(len(occupied) <= 1 and all(p.name in ('slot_0', 'slot_1') for p in occupied), 'Only two feature slots')
        folder.mkdir()
        write_new(folder / 'owner.json', owner)
    store = TransientStore(folder, owner, guard, lambda: engine._lock(folder.parent / 'slots.lock'))
    return record(folder / 'owner.json', guard), store, request, session, contract, validated, plan, normalized


def _clean_contract(row, expected, reference):
    ids = expected['full_target_ids']
    ceiling = 9223 if expected['role'] == 'student_tr' else 8193
    require(ids and all(type(v) is int and 0 <= v < ceiling and v != 8192 for v in ids), 'Full current clean/replay targets')
    return {'draw': deepcopy(expected['context']), 'role': expected['role'],
            'raw_text_sha256': expected['raw_text_sha256'], 'duration_metadata': row['duration'],
            'encoder_frames': row['encoder_frames'], 'target_tokens': len(ids),
            'target_ids_u32_sha256': expected['target_ids_u32_sha256'], 'target_vocabulary': ceiling,
            'clean_PCM': deepcopy(reference['expected_audio'])}


class SinglePassProducer:
    def __init__(self, owner_pin, store, normalized, *, clean_reader, weak_transform, frontend, guard):
        self.owner_pin, self.store, self.normalized = owner_pin, store, normalized
        self.clean_reader, self.weak_transform, self.frontend, self.guard = clean_reader, weak_transform, frontend, guard
        self.producer = identity(); self.parts = []; self.decode_count = 0

    def prepare_update(self, index):
        import numpy as np
        require(index == len(self.parts) and identity() == self.producer, 'Same serial producer and next update only')
        update = self.normalized[index]; entries = []; events = []; references = []
        for j, (raw, expected, context) in enumerate(zip(update['coordinator_update']['rows'], update['rows'],
                                                       update['coordinator_update']['contexts'])):
            self.guard(); row = deepcopy(raw); row['audio_draw_context'] = deepcopy(context)
            row['augmentation_role'] = expected['role']; original_row = digest(row)
            wave, reference = self.clean_reader(row, expected, update['phase'])
            self.decode_count += 1
            require(isinstance(wave, np.ndarray) and wave.dtype == np.float32 and wave.ndim == 1
                    and wave.flags.c_contiguous and np.isfinite(wave).all(), 'Full finite original float32 waveform')
            clean_sha = engine._wave_hash(wave, self.guard)
            require(reference['expected_audio'] == {'sample_rate': 16000, 'samples': len(wave), 'waveform_sha256': clean_sha}
                    and reference['full_target_ids'] == expected['full_target_ids']
                    and reference['context'] == context, 'Actual same-pass full PCM/target/context evidence')
            protected = update['phase'] == bridge.CLEANER or expected['role'] != 'student_tr' or len(wave) > 120 * 16000
            if update['phase'] == bridge.WEAK and expected['role'] == 'student_tr':
                prepared, augmentation = self.weak_transform(wave, row, deepcopy(context))
            else:
                prepared = wave
                augmentation = {'policy_sha256': self.store.owner['augmentation_identity']['policy_sha256'],
                                'source_id': row['source_id'], 'applied': False,
                                'label_policy': 'unchanged_single_target',
                                'reason': 'explicit_cleaner_phase_direct_original_audio' if update['phase'] == bridge.CLEANER
                                else 'protected_split_phase_role_or_language'}
            prepared_sha = engine._wave_hash(prepared, self.guard)
            require(prepared.dtype == np.float32 and prepared.ndim == 1 and prepared.flags.c_contiguous
                    and len(prepared) == len(wave) and np.isfinite(prepared).all()
                    and engine._wave_hash(wave, self.guard) == clean_sha
                    and augmentation['policy_sha256'] == self.store.owner['augmentation_identity']['policy_sha256']
                    and augmentation['source_id'] == row['source_id'] and augmentation['label_policy'] == 'unchanged_single_target'
                    and type(augmentation['applied']) is bool, 'Unchanged source/targets/full geometry and frozen transform')
            require(not protected or augmentation['applied'] is False and prepared_sha == clean_sha,
                    'Replay/teacher/cleaner/long clips remain full and clean')
            require(augmentation['applied'] or prepared_sha == clean_sha, 'Unapplied transform cannot change bytes')
            arrays = self.frontend(prepared)
            require(set(arrays) == {'input_features', 'attention_mask'} and digest(row) == original_row
                    and engine._wave_hash(wave, self.guard) == clean_sha
                    and engine._wave_hash(prepared, self.guard) == prepared_sha, 'Frontend must not mutate source or prepared PCM')
            entry = {k: deepcopy(expected[k]) for k in ('source_id', 'language', 'context', 'role', 'original_row_sha256',
                     'raw_text_sha256', 'full_target_ids', 'target_ids_u32_sha256')}
            entry.update(cleaner_original_ordinal=row.get('cleaner_original_ordinal'), target_tokens=len(expected['full_target_ids']),
                         encoder_frames=row['encoder_frames'], sample_rate=16000, samples=len(wave),
                         clean_PCM_sha256=clean_sha, prepared_PCM_sha256=prepared_sha)
            for key, suffix in (('input_features', 'features'), ('attention_mask', 'mask')):
                entry[key] = self.store.save('u%02d_r%02d_%s.npy' % (index, j, suffix), arrays[key])
            event = {'state': 'prepared_not_committed', 'batch_id': update['batch_id'], 'batch_row': j,
                     'draw': deepcopy(context), 'augmentation': augmentation}
            if update['phase'] == bridge.CLEANER:
                event.update(clean_protection=_clean_contract(row, expected, reference), clean_waveform_sha256=clean_sha)
            entries.append(entry); events.append(event); references.append(reference)
            del arrays, prepared, wave
            self.clean_reader.finish_row()
        payload = {'kind': 'FastMSS_prepared_batch_v1', 'state': 'prepared_not_committed', 'batch_id': update['batch_id'],
                   'rows': events, 'payload_sha256': digest(events), 'committed_frontier_advanced': False,
                   'commit_owner': 'native caller only after actual positive optimizer commit'}
        if update['phase'] == bridge.CLEANER:
            payload.update(protection_kind=engine.CLEAN_PROTECTION,
                           identity={'kind': engine.CLEAN_PROTECTION, 'wrapper': record(__file__, self.guard),
                                     'phase': bridge.CLEANER, 'frozen_broad_augmentation': self.store.owner['augmentation_identity'],
                                     'waveform_cache_created': False, 'augmentation_or_transport_invoked': False})
        original = update['coordinator_update']
        value = {'kind': UPDATE, 'owner': self.owner_pin, 'producer': self.producer, 'update_index': index,
                 'phase': update['phase'], 'rows': entries, 'prepared_payload': payload,
                 'same_pass_clean_evidence': references, 'independent_reference_process_claimed': False,
                 'prepared_is_committed': False, 'sampler_journal_frontier_advanced': False}
        value.update({k: deepcopy(original[k]) for k in ('global_update', 'batch_id', 'native_transaction_sha256',
                                                       'before_sampler_sha256', 'after_sampler_sha256')})
        part = self.store.write_metadata('update_%02d.json' % index, value)
        self.parts.append({'update': part})
        return part


def seal_actual_window(report_pin, actual_process0, manifest_output, *, guard, adoption=None):
    report = read_bound(report_pin, guard); request = read_bound(report['request'], guard)
    actual = read_bound(actual_process0, guard)
    verify(actual, kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis', report=report_pin, guard=guard)
    require(report['kind'] == REPORT and report['passed'] is True and report['producer'] == actual['producer']
            and actual['worker_source'] == report['worker_source'] == request['worker_source']
            and actual['parent_source'] == report['parent_source'] == request['parent_source']
            and report['actual_optimizer_updates'] == report['draws_consumed'] == 0
            and report['actual_decode_count'] == report['full_rows'], 'One genuinely closed single-decode numerical child')
    if request.get('provisional_predecessor') is not None:
        require(adoption is not None, 'A private future window is not yet consumable')
        accepted = read_bound(adoption, guard); pointer = read_bound(accepted['actual_predecessor_pointer'], guard)
        predicted = request['provisional_predecessor']
        require(accepted['kind'] == 'ASR18_singlepass_provisional_window_adoption_v1'
                and accepted['request'] == report['request'] and accepted['owner'] == report['owner']
                and pointer['committed_updates'] == predicted['expected_optimizer_updates']
                and pointer['sampler_state_sha256'] == predicted['expected_sampler_sha256']
                and accepted['successor_plan'] == request['plan'], 'Real successful predecessor joins the exact private successor')
    value = {'kind': MANIFEST, 'source': record(__file__, guard), 'owner': report['owner'], 'request': report['request'],
             'actual_CPU_process0': actual_process0, 'report': report_pin, 'updates': report['updates'],
             'adoption': adoption, 'prepared_is_committed': False, 'sampler_journal_frontier_advanced': False}
    target = Path(manifest_output).resolve(); require(target.is_relative_to(ROOT) and not target.exists(), 'Fresh durable manifest outside owned slot')
    require(not target.is_relative_to(exact_slot(request)), 'Manifest survives feature retirement')
    target.parent.mkdir(parents=True, exist_ok=True); write_new(target, value)
    return record(target, guard)


class FeatureWindowReader(engine._ReadCursor):
    def __init__(self, owner_pin, manifest_pin, *, guard):
        owner = read_bound(owner_pin, guard); manifest = read_bound(manifest_pin, guard)
        require(owner['kind'] == OWNER and owner['source'] == record(__file__, guard)
                and manifest['kind'] == MANIFEST and manifest['source'] == record(__file__, guard)
                and manifest['owner'] == owner_pin and manifest['request'] == owner['request'], 'Actual source-bound singlepass owner/closed manifest')
        request = read_bound(owner['request'], guard)
        require(Path(owner_pin['path']).resolve() == exact_slot(request) / 'owner.json', 'Exact stable slot owner')
        actual = read_bound(manifest['actual_CPU_process0'], guard)
        verify(actual, kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis', report=manifest['report'], guard=guard)
        report = read_bound(manifest['report'], guard)
        require(report['owner'] == owner_pin and report['updates'] == manifest['updates']
                and report['worker_source'] == request['worker_source'] == actual['worker_source']
                and report['producer'] == actual['producer'] and report['passed'] is True,
                'Actual numerical child0, never a metadata fixture')
        if request.get('provisional_predecessor') is not None:
            require(manifest['adoption'] is not None, 'Future private arrays require actual predecessor adoption')
            adoption = read_bound(manifest['adoption'], guard); pointer = read_bound(adoption['actual_predecessor_pointer'], guard)
            require(adoption['request'] == owner['request'] and pointer['committed_updates'] == request['provisional_predecessor']['expected_optimizer_updates']
                    and pointer['sampler_state_sha256'] == request['provisional_predecessor']['expected_sampler_sha256'], 'Actual future frontier before native consumption')
        plan = read_bound(request['plan'], guard)
        normalized = bridge.normalize_updates(plan, population=bridge.validate_DATA(request['DATA_evidence'], guard)['retained_rows'], guard=guard)
        require(len(manifest['updates']) == owner['updates'] == len(normalized), 'Every planned update present exactly once')
        for i, (part, expected_update) in enumerate(zip(manifest['updates'], normalized)):
            require(Path(part['update']['path']).resolve() == exact_slot(request) / ('update_%02d.json' % i), 'Exact owned ordered update')
            value = read_bound(part['update'], guard); original = expected_update['coordinator_update']
            require(value['kind'] == UPDATE and value['owner'] == owner_pin and value['producer'] == report['producer']
                    and value['update_index'] == i and value['phase'] == expected_update['phase']
                    and value['prepared_is_committed'] is False and value['sampler_journal_frontier_advanced'] is False
                    and len(value['rows']) == len(expected_update['rows']) == len(value['same_pass_clean_evidence'])
                    and all(value[k] == original[k] for k in ('global_update', 'batch_id', 'native_transaction_sha256', 'before_sampler_sha256', 'after_sampler_sha256')),
                    'Full original plan, native transaction, source frontier and actual producer')
            payload = value['prepared_payload']
            require(payload['state'] == 'prepared_not_committed' and payload['committed_frontier_advanced'] is False
                    and payload['batch_id'] == original['batch_id'] and payload['payload_sha256'] == digest(payload['rows'])
                    and len(payload['rows']) == len(value['rows']), 'Exact once-consumed immutable augmentation payload')
            for j, (entry, expected, reference, event) in enumerate(zip(value['rows'], expected_update['rows'], value['same_pass_clean_evidence'], payload['rows'])):
                require(all(entry[k] == expected[k] for k in ('source_id','language','context','role','original_row_sha256','raw_text_sha256','full_target_ids','target_ids_u32_sha256'))
                        and entry['target_tokens'] == len(expected['full_target_ids']) and reference['full_target_ids'] == entry['full_target_ids']
                        and reference['context'] == entry['context'] and reference['whole_original_clip'] is True
                        and reference['actual_geometry_and_heldout_checked'] is True and reference['augmentation_applied'] is False
                        and reference['expected_audio'] == {'sample_rate':16000,'samples':entry['samples'],'waveform_sha256':entry['clean_PCM_sha256']}
                        and event['batch_row'] == j and event['batch_id'] == original['batch_id'] and event['draw'] == entry['context'],
                        'Full labels, actual clean PCM, geometry, heldout and frozen ordered draw before native consumption')
                raw = original['rows'][j]
                require(entry['encoder_frames'] == raw['encoder_frames'] == ((entry['samples'] // 160 + 7) // 8), 'Original valid-mask encoder geometry')
                for key, suffix in (('input_features','features'),('attention_mask','mask')):
                    require(Path(entry[key]['file']['path']).resolve() == exact_slot(request) / ('u%02d_r%02d_%s.npy' % (i,j,suffix)), 'Exact owned array source')
                aug = event['augmentation']
                require(aug['policy_sha256'] == owner['augmentation_identity']['policy_sha256'] and aug['source_id'] == entry['source_id']
                        and type(aug['applied']) is bool and aug['label_policy'] == 'unchanged_single_target', 'Frozen augmentation policy and labels')
                if expected_update['phase'] == bridge.CLEANER or expected['role'] != 'student_tr' or entry['samples'] > 120*16000:
                    require(aug['applied'] is False and entry['clean_PCM_sha256'] == entry['prepared_PCM_sha256'], 'Protected sources remain clean')
        self.manifest, self.owner = manifest, owner
        store = engine._Store(exact_slot(request), owner, guard, lambda: engine._lock(exact_slot(request).parent / 'slots.lock'))
        super().__init__(store, lambda i: manifest['updates'][i]['update'], updates=owner['updates'], guard=guard)


def cleanup_window(owner_pin, manifest_pin, *, archive_path, actual_boundary, verify_boundary, guard):
    owner = read_bound(owner_pin, guard); request = read_bound(owner['request'], guard)
    require(callable(verify_boundary) and verify_boundary(actual_boundary) is True, 'Actual caller-verified coherent consumed boundary')
    pointer = read_bound(actual_boundary['coherent_checkpoint_pointer'], guard)
    plan = read_bound(request['plan'], guard)
    require(pointer['all_window_updates_committed'] is True
            and pointer['sampler_state_sha256'] == digest(plan['final_private_planning_sampler'])
            and actual_boundary['owner'] == owner_pin and actual_boundary['manifest'] == manifest_pin,
            'Only this actually consumed window may retire')
    folder = exact_slot(request); store = engine._Store(folder, owner, guard, lambda: engine._lock(folder.parent / 'slots.lock'))
    guard(); store.inventory(); files = list(folder.iterdir())
    require(set(p.name for p in files) <= set(owner['owned_names']), 'Unknown file preserved')
    archive = {'kind': 'FastMSS_singlepass_owned_window_durable_archive_v1', 'owner': owner_pin,
               'manifest': manifest_pin, 'actual_boundary': actual_boundary, 'files': {p.name: record(p, guard) for p in files},
               'metadata': {p.name: json.loads(p.read_bytes()) for p in files if p.suffix == '.json'}}
    target = Path(archive_path).resolve(); require(target.is_relative_to(ROOT) and not target.is_relative_to(folder) and not target.exists(), 'Fresh archive outside exact owned slot')
    target.parent.mkdir(parents=True, exist_ok=True); write_new(target, archive); saved = record(target, guard)
    with store.lock():
        require({p.name: record(p, guard) for p in folder.iterdir()} == archive['files'] and record(target, guard) == saved,
                'Complete archive/hash authority before owned transient unlink')
        for p in files: require(p.resolve().parent == folder and p.name in owner['owned_names'], 'Owned-only unlink'); p.unlink()
        folder.rmdir()
    return {'passed': True, 'archive': saved, 'owned_files_removed': len(files), 'originals_checkpoints_journals_untouched': True}


def discard_private_window(owner_pin, report_pin, actual_process0, *, archive_path, private_boundary, verify_private, guard):
    """Retire a closed CPU-only measurement; never pretend it was consumed."""
    require(callable(verify_private) and verify_private(private_boundary) is True
            and private_boundary['kind'] == 'ASR18_unconsumed_singlepass_CPU_measurement_boundary_v1'
            and private_boundary['actual_native_updates'] == private_boundary['actual_draws_consumed'] == 0
            and private_boundary['actual_CPU_process0'] == actual_process0,
            'Owner-observed real CPU0 and no native consumption required')
    owner = read_bound(owner_pin, guard); request = read_bound(owner['request'], guard)
    report = read_bound(report_pin, guard); actual = read_bound(actual_process0, guard)
    verify(actual, kind='ASR18_owned_CPU_process_completion_v1', owner='/root/memory_diagnosis', report=report_pin, guard=guard)
    require(report['owner'] == owner_pin and report['passed'] is True and report['request'] == owner['request']
            and report['actual_optimizer_updates'] == report['draws_consumed'] == 0
            and report['producer'] == actual['producer'] and actual['worker_source'] == request['worker_source'],
            'Exact successful private CPU producer; no state transition')
    folder = exact_slot(request); store = engine._Store(folder, owner, guard, lambda: engine._lock(folder.parent / 'slots.lock'))
    guard(); store.inventory(); files = list(folder.iterdir())
    require(set(p.name for p in files) <= set(owner['owned_names']), 'Preserve unknown transient files')
    archive = {'kind':'FastMSS_unconsumed_singlepass_CPU_measurement_archive_v1', 'owner':owner_pin,
               'report':report_pin, 'actual_CPU_process0':actual_process0, 'private_boundary':private_boundary,
               'files':{p.name:record(p,guard) for p in files},
               'metadata':{p.name:json.loads(p.read_bytes()) for p in files if p.suffix=='.json'},
               'prepared_rows_consumed':False, 'authoritative_state_advanced':False}
    target=Path(archive_path).resolve()
    require(target.is_relative_to(ROOT) and not target.is_relative_to(folder) and not target.exists(), 'Fresh archive outside owned transient slot')
    target.parent.mkdir(parents=True,exist_ok=True); write_new(target,archive); saved=record(target,guard)
    with store.lock():
        require({p.name:record(p,guard) for p in folder.iterdir()}==archive['files'] and record(target,guard)==saved,
                'Complete private measurement archive before owned-only retirement')
        for p in files:
            require(p.resolve().parent==folder and p.name in owner['owned_names'],'Exact owned transient only');p.unlink()
        folder.rmdir()
    return {'passed':True,'archive':saved,'owned_files_removed':len(files),'native_updates':0,'draws_consumed':0}
