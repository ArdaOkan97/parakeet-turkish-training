"""Finite original CPU frontend window; private hash-only validation first.

No NPY/PCM/shared-owner files are written. The full original frontend and row
protocol execute on real originals; output hashes compare against an actually
closed CPU8 oracle. One aggregate exit is required, never per-update child0.
"""
import argparse
import gc
import hashlib
import inspect
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastmss_row_serial_cpu_v8 as core
import produce_FastMSS_reference_window_CPU_v1 as reference

KIND = 'FastMSS_private_hash_only_original_frontend_window_v1'


class HashOnlyStore:
    """Private comparison engine; no manufactured NPY or public owner pins."""
    def __init__(self, index, oracle_rows, np, guard, timings):
        self.index, self.oracle, self.np, self.guard, self.timings = index, oracle_rows, np, guard, timings
        self.pending = {}
        self.completed = 0
        self.expected = {}
        for row, entry in enumerate(oracle_rows):
            for key, suffix in (('input_features', 'features'), ('attention_mask', 'mask')):
                name = f'u{index:02}_r{row:02}_{suffix}.npy'
                core.require(Path(entry[key]['file']['path']).name == name, 'Exact closed original array name')
                self.expected[name] = (row, key, entry[key])

    def save(self, name, value):
        with self.timings.stage('feature_mask_hash_comparison_no_file_write'):
            self.guard()
            core.require(name in self.expected and name not in self.pending, 'Exact next private hash-only member')
            row, key, original = self.expected[name]
            core.require(row == self.completed and value.flags.c_contiguous
                and str(value.dtype) == original['dtype'] and list(value.shape) == original['shape'],
                'Actual complete original feature dtype/physical shape/layout')
            sha = hashlib.sha256(memoryview(value).cast('B')).hexdigest()
            core.require(sha == original['array_sha256'], 'Full actual frontend bytes differ from closed original')
            self.pending[name] = value
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'array_sha256': sha,
                    'readonly_oracle_file': original['file'], 'new_file_written': False}

    def arrays(self, entry):
        with self.timings.stage('original_full_mask_finite_geometry_validation'):
            np = self.np
            f = f'u{self.index:02}_r{self.completed:02}_features.npy'
            m = f'u{self.index:02}_r{self.completed:02}_mask.npy'
            core.require(set(self.pending) == {f, m}, 'Two complete arrays from exactly one current row')
            features, mask = self.pending[f], self.pending[m]
            core.require(features.ndim == 3 and features.shape[0] == 1 and features.shape[2] == 128
                and mask.shape == features.shape[:2] and mask.any() and np.isfinite(features).all()
                and np.array_equal(mask, np.arange(mask.shape[1])[None, :] < mask.sum(1)[:, None])
                and not features[~mask].any() and (int(mask.sum()) + 7) // 8 == entry['encoder_frames'],
                'All original finite/right-mask/zero-pad/full encoder checks')
            self.pending.clear(); self.completed += 1


def main(request_path, output):
    began = time.monotonic(); core.environment(); timings = reference.Timings()
    with timings.stage('immutable_metadata_and_oracle_validation'):
        request_pin, request, plan, contract, validated, oracle = reference.load_window_request(request_path, 'frontend')
        reference_window, reference_updates = reference.closed_reference_window(request, request_pin, plan, core.resource_guard)
    core.require(core.same(request['sources']['frontend'], core.record(__file__)), 'Actual new frontend-window source')
    guard = core.stage_guard(contract, 'frontend')
    admission = core.admit(contract['resource_review']['frontend_incremental_bytes'], guard)
    folder = reference.output_folder(request, output, 'frontend')
    reader = None; froze = False; updates = []; timings.install_gc_timer()
    try:
        with timings.stage('NumPy_CPU_Torch_Transformers_import_and_extractor_setup'):
            np, torch = core.numpy_runtime(frontend=True)
            import scipy
            identity = core.bridge.frozen_augmentation(plan['augmentation_identity'], guard)
            core.require(np.__version__ == identity['numpy_version'] and scipy.__version__ == identity['scipy_version'],
                         'Exact unchanged augmentation runtime identity')
            from transformers import AutoFeatureExtractor
            declared = contract['frontend']
            extractor = AutoFeatureExtractor.from_pretrained(declared['base_path'], local_files_only=True)
            core.require(type(extractor).__name__ == 'ParakeetFeatureExtractor'
                and core.same(core.record(inspect.getfile(type(extractor)), guard), declared['source']),
                'Exact original CPU frontend, no model construction')
        with timings.stage('reader_augmentation_protocol_setup'):
            reader = core.OriginalRows(validated, raw_contract=contract['raw_lifetime_review'],
                private_limit_bytes=contract['resource_review']['frontend_maximum_private_bytes'],
                frontend_enabled=True, guard=guard)
            weak_transform = None
            if plan['phase'] == core.bridge.WEAK:
                import fastmss_train_augmentation_v1 as adapter
                from fastmss_ordered_draw_binding_v1 import FastMSSOrderedDrawBinding, AugmentationDrawContext
                augmenter = adapter.TrainWaveformAugmenter(adapter.AugmentationPolicy.from_json(identity['policy_file']['path']), guard=guard)
                binding = FastMSSOrderedDrawBinding(augmenter, policy_path=identity['policy_file']['path'], adapter_path=adapter.__file__)
                core.require(core.same(binding.identity, identity), 'Frozen Gaussian/twoRIR pilot only, no new asset activation')
                def weak_transform(wave, row, context):
                    with timings.stage('original_weak_transform'):
                        return reader.transform(lambda w, r, c: augmenter.apply(w, r, **AugmentationDrawContext(**c).adapter_context()), wave, row, context)
            active = {'index': None, 'cursor': 0, 'geometry': []}
            def clean_reader(row):
                index, cursor = active['index'], active['cursor']
                update = plan['updates'][index]
                core.require(cursor < len(update['rows'])
                    and core.same(row['audio_draw_context'], update['rows'][cursor]['context']), 'Exact original row/order/context')
                with timings.stage('original_decode_resample_hash_geometry_heldout'):
                    wave = reader(row, update['rows'][cursor], update['phase'])
                    expected = reference_updates[index]['references'][cursor]
                    core.require(core.same(reader.last, expected['actual_geometry'])
                        and core.same(reader.geometry_proof, expected['before_draw_geometry_proof'])
                        and core.same(reader.original_source_files, expected['original_source_files']),
                        'Independent complete native/16k/PCM/source/heldout proof must reproduce')
                    if update['rows'][cursor]['role'] == 'replay':
                        core.require(core.same(reader.replay_proof, expected['genuine_teacher_feature_cache']), 'Exact genuine replay teacher features')
                active['cursor'] += 1
                return wave
            def original_frontend(wave):
                with timings.stage('original_Parakeet_CPU_frontend'):
                    guard(); inputs = dict(extractor([wave], **core.sink.FRONTEND_ARGS))
                    core.require(set(inputs) == {'input_features', 'attention_mask'}, 'Complete original frontend')
                    mask = inputs['attention_mask'].bool()
                    features = inputs['input_features'].masked_fill(~mask.unsqueeze(-1), 0).detach().cpu().contiguous().numpy()
                    arrays = {'input_features': features, 'attention_mask': mask.detach().cpu().contiguous().numpy()}
                    active['geometry'].append(core.check_frontend(arrays, len(wave), np))
                    return arrays
            def frontend(wave):
                return reader.frontend(original_frontend, wave)
            stores, protocols = [], []
            for index, update in enumerate(plan['updates']):
                refs = reference_updates[index]['references']
                core.require(core.same(refs, oracle[index]['references']), 'Actually closed new references equal closed original references')
                store = HashOnlyStore(index, oracle[index]['feature_update']['rows'], np, guard, timings)
                stores.append(store)
                protocols.append(core.sink._RowProtocol(update, refs, identity, store,
                    clean_reader=clean_reader, weak_transform=weak_transform, transform_identity=identity, frontend=frontend, guard=guard))
        core.require(reader.active_job is None and not reader.ledger.jobs and gc.get_freeze_count() == 0,
                     'One fresh quiescent bootstrap before any waveform/feature graph')
        freeze_count = 0
        if request['freeze_after_setup']:
            with timings.stage('bootstrap_collect_and_freeze'):
                gc.collect(); gc.freeze(); froze = True; freeze_count = gc.get_freeze_count()
        for index, (update, producer, store) in enumerate(zip(plan['updates'], protocols, stores)):
            guard(); core.require(reader.active_job is None and not reader.ledger.jobs, 'Released raw row/update boundary')
            active.update(index=index, cursor=0, geometry=[])
            for row in range(len(update['rows'])):
                with timings.stage('complete_original_row_protocol'):
                    producer.prepare_row(row)
                with timings.stage('outer_row_gc'):
                    gc.collect()
                with timings.stage('raw_weakref_finish_including_original_inner_gc'):
                    reader.finish_raw_row()
            rows, payload = producer.prepared()
            original = oracle[index]['feature_update']
            core.require(store.completed == len(update['rows']) and not store.pending
                and active['cursor'] == len(update['rows']) == len(active['geometry'])
                and core.same(payload, original['prepared_payload']), 'All complete rows and exact original copied payload')
            for actual, expected in zip(rows, original['rows']):
                core.require(set(actual) == set(expected)
                    and all(core.same(actual[k], expected[k]) for k in expected if k not in ('input_features', 'attention_mask')),
                    'All immutable full target/context/source/clean/prepared PCM metadata must match')
            target = folder / f'update_{index:02}.json'
            core.write_new(target, {'kind': 'FastMSS_private_full_frontend_update_within_live_window_v1',
                'passed': True, 'producer': core.identity(), 'update_index': index, 'bridge': request['bridge'],
                'rows': rows, 'prepared_payload': payload, 'actual_geometry': active['geometry'],
                'baseline_feature_update': oracle[index]['feature_update_pin'],
                'new_NPY_files_written': 0, 'per_update_child0_claimed': False, 'native_features_ready': False})
            updates.append({'update_index': index, 'report': core.record(target, guard)})
        receipt = reader.raw_receipt()
    finally:
        if froze:
            gc.unfreeze(); gc.collect()
        if reader is not None: reader.close()
        timings.close()
    core.require(not any(k == 'pyarrow' or k.startswith('pyarrow.') or k.startswith('nemo.collections.asr.models') for k in sys.modules),
                 'No Arrow/ASR model load')
    result = {'kind': KIND, 'mode': reference.MODE, 'passed': True, 'producer': core.identity(),
        'worker_source': core.record(__file__, guard), 'parent_source': request['parent_source'],
        'request': request_pin, 'bridge': request['bridge'], 'reference_window_report': request['reference_window_report'],
        'reference_window_actual0': request['reference_window_actual0'], 'updates': updates,
        'all_updates_reported': True, 'full_rows': sum(len(v['rows']) for v in plan['updates']),
        'individual_array_raw_hashes': 2 * sum(len(v['rows']) for v in plan['updates']),
        'raw_lifetime_receipt': receipt, 'raw_PCM_files_written': 0, 'new_NPY_files_written': 0,
        'native_features_ready': False, 'public_owner_constructor_used': False,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'actual_process_exit_observed_in_child': False, 'actual_aggregate_exit_required': True,
        'freeze_after_setup': request['freeze_after_setup'], 'bootstrap_frozen_objects': freeze_count,
        'unfrozen_before_reader_teardown': True, 'timings': timings.values, 'gc_events': timings.gc_events,
        'NumPy': np.__version__, 'Torch': torch.__version__, 'SciPy': scipy.__version__, 'torch_threads': torch.get_num_threads(),
        'CPU_stage_admission': admission, 'wall_seconds': time.monotonic() - began,
        'array_write_fsync_readback_and_inventory_omitted_from_private_timing': True,
        'actual_full_CPU_preparation_speedup_claimed': False, 'production_enabled': False}
    core.write_new(core.scoped(output), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.request, args.output)
