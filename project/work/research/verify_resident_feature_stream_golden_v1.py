"""Actual original-audio RAM transport versus the sealed p13 numerical output."""
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / 'outputs/parakeet_turkish'), str(PROJECT / 'work/research')]
import fastmss_BPE9223_cpu_common_v12 as core
import fastmss_resident_feature_stream_v1 as stream


def main(output):
    core.environment()
    np, _ = core.numpy_runtime(frontend=False)
    area = stream.ROOT / 'analysis/BPE9223_training_launch_v1'
    golden = stream.bound(area / 'singlepass_CPU_p13_benchmark_actual_v1/retained_CPU_measurement_feature_metadata.json')
    report = stream.bound(area / 'singlepass_CPU_p13_benchmark_actual_v1/report.json')
    request = stream.bound(report['request'])
    plan_pin = request['plan']
    plan = stream.bound(plan_pin)
    config = stream.pin(area / 'pipeline_overlap_configuration_v2/configuration_pending_execution_review.json')
    output = Path(output).resolve()
    stream.require(output.is_relative_to(stream.ROOT) and not output.exists(), 'Fresh E fixture output')
    output.mkdir(parents=True)
    began = time.monotonic()
    child = stream.ResidentFeatureProducer(config, output / 'producer', queue_bytes=stream.DEFAULT_QUEUE)
    checks = {'full_rows': 0, 'exact_array_SHA': 0, 'exact_PCM_targets_contexts': 0, 'exact_augmentation_payloads': 0}
    times = []
    try:
        sha = child.submit(plan_pin)
        for i in range(len(plan['updates'])):
            packet = child.next_update(sha, i)
            expected = golden['metadata']['update_%02d.json' % i]
            actual = packet['update']
            stream.require(actual['prepared_payload'] == expected['prepared_payload'], 'Original deterministic noise/room payload')
            checks['exact_augmentation_payloads'] += 1
            for old, new, arrays in zip(expected['rows'], actual['rows'], packet['arrays']):
                for key in ('input_features', 'attention_mask'):
                    array = arrays[key]
                    digest = hashlib.sha256(memoryview(array).cast('B')).hexdigest()
                    stream.require(old[key]['array_sha256'] == new[key]['array_sha256'] == digest and
                                   old[key]['shape'] == list(array.shape) and old[key]['dtype'] == str(array.dtype),
                                   'Every full feature/mask byte and original geometry')
                    checks['exact_array_SHA'] += 1
                keys = set(old) - {'input_features', 'attention_mask'}
                stream.require({k: old[k] for k in keys} == {k: new[k] for k in keys},
                               'Full original PCM, targets, context, geometry and source identity')
                checks['exact_PCM_targets_contexts'] += 1
                checks['full_rows'] += 1
            times.append(packet['timings']['preparation_seconds'])
            child.release(packet['token'])
            del packet, arrays, actual
        closure = child.close()
        result = {'kind': 'FastMSS_actual_resident_RAM_golden_comparison_v1', 'passed': True,
                  **checks, 'wall_seconds': time.monotonic() - began,
                  'sum_preparation_seconds': sum(times), 'per_update_preparation_seconds': times,
                  'CPU_closure': closure, 'source': stream.pin(__file__),
                  'actual_native_updates': 0, 'authoritative_draws_consumed': 0,
                  'real_data_and_recorded_noise_16rooms': True,
                  'feature_files_created': 0, 'unconditional_per_row_gc_removed': True}
        (output / 'actual_summary.json').write_bytes(stream.canonical(result))
        print(stream.canonical({k: v for k, v in result.items() if k != 'CPU_closure'}).decode(), flush=True)
    except BaseException:
        child.abort()
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
