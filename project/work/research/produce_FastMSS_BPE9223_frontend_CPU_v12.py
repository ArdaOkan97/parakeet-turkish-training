"""Full original CPU frontend window; real NPY sink7 and aggregate source protocol.

NEW isolated owned NPY files are written; no PCM or old shared-owner mutation. The full original frontend and row
protocol execute on real originals, with the unchanged pinned numerical frontend.
Independent whole-clean-PCM evidence closes before this leaf begins. One actual
aggregate exit is required, never a per-update child0.
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
import fastmss_BPE9223_cpu_common_v12 as core
import produce_FastMSS_BPE9223_reference_CPU_v12 as reference
import fastmss_BPE9223_feature_protocol_v12 as public

KIND = public.FRONTEND


def main(request_path, output):
    began = time.monotonic(); core.environment(); timings = reference.Timings()
    with timings.stage('immutable_metadata_and_oracle_validation'):
        request_pin, request, plan, contract, validated, oracle = reference.load_window_request(request_path, 'frontend')
        reference_window, reference_updates = reference.closed_reference_window(request, request_pin, plan, core.resource_guard)
    source, _ = public.verify_source_contract(request['source_contract'], guard=core.resource_guard)
    core.require(core.same(source['sources']['frontend'], core.record(__file__)), 'Actual new full frontend-window source')
    guard = core.stage_guard(contract, 'frontend')
    admission = core.admit(contract['resource_review']['frontend_incremental_bytes'], guard)
    folder = reference.output_folder(request, output, 'frontend')
    reader = None; froze = False; updates = []; timings.install_gc_timer()
    try:
        with timings.stage('NumPy_CPU_Torch_Transformers_import_and_extractor_setup'):
            np, torch = core.numpy_runtime(frontend=True)
            import scipy
            import fastmss_BPE9223_recorded_room_train_v3 as augmentation
            _, identity = augmentation.policy_identity(source['augmentation_policy'], guard)
            core.require(np.__version__ == source['numerical_runtime_versions']['NumPy'] and scipy.__version__ == source['numerical_runtime_versions']['SciPy'],
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
            augmenter = augmentation.RecordedRoomTrainingAdapter(source['augmentation_policy'], reader=reader,
                guard=guard, ROOT_review=request['ROOT_review'], configuration=request['configuration'])
            core.require(core.same(augmenter.identity, source['augmentation_identity']), 'Exact new reviewed draw policy')
            def weak_transform(wave, row, context):
                with timings.stage('recorded_noise_and_16room_training_transform'):
                    return augmenter.transform(wave, row, context)
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
                return augmenter.frontend(original_frontend, wave)
            producer = public.WindowProducer(request['owner'], reference_report=request['reference_window_report'],
                reference_actual0=request['reference_window_actual0'], clean_reader=clean_reader,
                weak_transform=weak_transform, transform_identity=identity, frontend=frontend, guard=guard)
            original_save, original_arrays, original_inventory = producer.store.save, producer.store.arrays, producer.store.inventory
            def save(name, array):
                with timings.stage('actual_NPY_write_fsync_file_and_array_hash'):
                    return original_save(name, array)
            def arrays(entry):
                with timings.stage('actual_NPY_full_readback_and_array_validation'):
                    return original_arrays(entry)
            def inventory():
                with timings.stage('actual_full_owned_inventory_scans'):
                    return original_inventory()
            producer.store.save, producer.store.arrays, producer.store.inventory = save, arrays, inventory
        core.require(reader.active_job is None and not reader.ledger.jobs and gc.get_freeze_count() == 0,
                     'One fresh quiescent bootstrap before any waveform/feature graph')
        freeze_count = 0
        if request['freeze_after_setup']:
            with timings.stage('bootstrap_collect_and_freeze'):
                gc.collect(); gc.freeze(); froze = True; freeze_count = gc.get_freeze_count()
        for index, update in enumerate(plan['updates']):
            protocol = producer.begin_update(index)
            guard(); core.require(reader.active_job is None and not reader.ledger.jobs, 'Released raw row/update boundary')
            active.update(index=index, cursor=0, geometry=[])
            for row in range(len(update['rows'])):
                with timings.stage('complete_original_row_protocol'):
                    protocol.prepare_row(row)
                with timings.stage('outer_row_gc'):
                    gc.collect()
                with timings.stage('raw_weakref_finish_including_original_inner_gc'):
                    reader.finish_raw_row()
            core.require(active['cursor'] == len(update['rows']) == len(active['geometry']), 'Complete actual frontend row/geometry count')
            producer.finish_update(index)
        updates = producer.prepared()
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
        'request': request_pin, 'source_contract': request['source_contract'], 'owner': request['owner'], 'bridge': request['bridge'], 'reference_window_report': request['reference_window_report'],
        'reference_window_actual0': request['reference_window_actual0'], 'updates': updates,
        'all_updates_reported': True, 'per_update_child0_claimed': False, 'full_rows': sum(len(v['rows']) for v in plan['updates']),
        'individual_array_raw_hashes': 2 * sum(len(v['rows']) for v in plan['updates']),
        'raw_lifetime_receipt': receipt, 'raw_PCM_files_written': 0, 'new_NPY_files_written': 2 * sum(len(v['rows']) for v in plan['updates']),
        'native_features_ready': False, 'public_owner_constructor_used': True,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'actual_process_exit_observed_in_child': False, 'actual_aggregate_exit_required': True,
        'freeze_after_setup': request['freeze_after_setup'], 'bootstrap_frozen_objects': freeze_count,
        'unfrozen_before_reader_teardown': True, 'timings': timings.values, 'gc_events': timings.gc_events,
        'NumPy': np.__version__, 'Torch': torch.__version__, 'SciPy': scipy.__version__, 'torch_threads': torch.get_num_threads(),
        'CPU_stage_admission': admission, 'wall_seconds': time.monotonic() - began,
        'array_write_fsync_readback_and_inventory_omitted_from_private_timing': False,
        'actual_full_CPU_preparation_speedup_claimed': False, 'production_enabled': False,
        'augmentation_identity': identity, 'native_benchmark_only': False}
    core.write_new(core.scoped(output), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.request, args.output)
