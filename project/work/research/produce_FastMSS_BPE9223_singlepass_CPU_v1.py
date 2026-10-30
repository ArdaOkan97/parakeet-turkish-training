"""One bounded real-audio decode/augmentation/frontend pass; CPU-only worker."""
import argparse
import gc
import inspect
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(P / 'outputs/parakeet_turkish'), str(P / 'work/research')]
import fastmss_singlepass_CPU_v1 as single
import fastmss_BPE9223_cpu_common_v12 as core
import produce_FastMSS_BPE9223_reference_CPU_v12 as timing_helpers


def main(request_path, output):
    core.environment(); began = time.monotonic()
    timings = timing_helpers.Timings(); request_pin = single.record(request_path, core.resource_guard)
    parsed = single.validate_request(request_pin, core.resource_guard)
    request, session, contract, validated, plan, normalized = parsed
    single.require(request['worker_source'] == single.record(__file__, core.resource_guard), 'Actual singlepass producer source')
    output = Path(output).resolve(); slot = single.exact_slot(request)
    single.require(output.is_relative_to(single.ROOT) and not output.is_relative_to(slot)
                   and not output.exists(), 'Fresh external worker report survives transient cleanup')
    guard = core.stage_guard(contract, 'frontend')
    admission = core.admit(contract['resource_review']['frontend_incremental_bytes'], guard)
    reader = None; froze = False
    with timings.stage('numerical_import_and_frontend_setup'):
        np, torch = core.numpy_runtime(frontend=True)
        import scipy
        import fastmss_BPE9223_recorded_room_train_v3 as augmentation
        from transformers import AutoFeatureExtractor
        _, policy_identity = augmentation.policy_identity(request['augmentation_policy'], guard)
        single.require(policy_identity == request['augmentation_identity'], 'Exact existing recorded noise/RIR policy')
        declared = request['frontend']
        extractor = AutoFeatureExtractor.from_pretrained(declared['base_path'], local_files_only=True)
        single.require(type(extractor).__name__ == 'ParakeetFeatureExtractor'
                       and core.record(inspect.getfile(type(extractor)), guard) == declared['source'],
                       'Original numerical frontend implementation')
    owner_pin, store, _, _, _, _, _, _ = single.create_window(request_pin, declared, guard, validated_inputs=parsed)
    try:
        reader = core.OriginalRows(validated, raw_contract=contract['raw_lifetime_review'],
            private_limit_bytes=contract['resource_review']['frontend_maximum_private_bytes'],
            frontend_enabled=True, guard=guard)
        augmenter = augmentation.RecordedRoomTrainingAdapter(request['augmentation_policy'], reader=reader,
            guard=guard, ROOT_review=request['augmentation_ROOT_review'], configuration=request['augmentation_configuration'])
        single.require(augmenter.identity == request['augmentation_identity'], 'Unchanged full deterministic training draw policy')
        def clean_reader(row, expected, phase):
            with timings.stage('one_original_decode_resample_hash_geometry_heldout'):
                wave = reader(row, expected, phase)
                evidence = core.reference_entry(row, expected, reader)
                return wave, evidence
        clean_reader.finish_row = reader.finish_raw_row
        def weak_transform(wave, row, context):
            with timings.stage('recorded_noise_and_room_transform'):
                return augmenter.transform(wave, row, context)
        def original_frontend(wave):
            with timings.stage('original_Parakeet_frontend'):
                guard(); inputs = dict(extractor([wave], **core.sink.FRONTEND_ARGS))
                single.require(set(inputs) == {'input_features', 'attention_mask'}, 'Complete original frontend output')
                mask = inputs['attention_mask'].bool()
                arrays = {'input_features': inputs['input_features'].masked_fill(~mask.unsqueeze(-1), 0).detach().cpu().contiguous().numpy(),
                          'attention_mask': mask.detach().cpu().contiguous().numpy()}
                core.check_frontend(arrays, len(wave), np)
                return arrays
        def frontend(wave):
            return augmenter.frontend(original_frontend, wave)
        producer = single.SinglePassProducer(owner_pin, store, normalized, clean_reader=clean_reader,
                                             weak_transform=weak_transform, frontend=frontend, guard=guard)
        single.require(reader.active_job is None and not reader.ledger.jobs, 'Raw ledger empty before numerical loop')
        gc.collect(); gc.freeze(); froze = True
        with timings.stage('complete_singlepass_window'):
            for index in range(len(normalized)):
                producer.prepare_update(index)
                guard(); single.require(reader.active_job is None and not reader.ledger.jobs, 'Released full raw row lifetimes')
        raw_receipt = reader.raw_receipt()
    finally:
        if froze: gc.unfreeze(); gc.collect()
        if reader is not None: reader.close()
    single.require(producer.decode_count == sum(len(v['rows']) for v in normalized)
                   and not any(k == 'pyarrow' or k.startswith('nemo.collections.asr.models') for k in sys.modules),
                   'Exactly one actual full decode per row, no Arrow/ASR model')
    report = {'kind': single.REPORT, 'passed': True, 'producer': single.identity(),
              'worker_source': single.record(__file__, guard), 'parent_source': request['parent_source'],
              'request': request_pin, 'owner': owner_pin, 'updates': producer.parts,
              'window_index': request['window_index'],
              'actual_optimizer_updates': 0, 'draws_consumed': 0, 'sampler_journal_frontier_advanced': False,
              'full_rows': producer.decode_count, 'actual_decode_count': producer.decode_count,
              'independent_reference_process_claimed': False, 'one_actual_combined_CPU_exit_required': True,
              'raw_lifetime_receipt': raw_receipt, 'CPU_stage_admission': admission,
              'NumPy': np.__version__, 'Torch': torch.__version__, 'SciPy': scipy.__version__,
              'torch_threads': torch.get_num_threads(), 'timings': timings.values,
              'wall_seconds': time.monotonic() - began, 'transient_per_array_fsync': False,
              'checkpoint_durability_policy_changed': False, 'production_enabled': False}
    output.parent.mkdir(parents=True, exist_ok=True); single.write_new(output, report)
    print(single.canonical({'report': single.record(output, guard), 'wall_seconds': report['wall_seconds']}).decode(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    arguments = parser.parse_args(); main(arguments.request, arguments.output)
