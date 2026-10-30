"""Publish the exact fail-closed compressed-carrier coverage receipt interface."""
from pathlib import Path
import datetime as dt
import hashlib
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

AREA = Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1')


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def main():
    binding = obj({'path': {'type': 'string'}, 'bytes': {'type': 'integer', 'minimum': 0},
                   'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                   'mtime_ns': {'type': 'integer', 'minimum': 0}})
    signature = obj({'path': {'type': 'string'}, 'bytes': {'type': 'integer', 'minimum': 0},
                     'mtime_ns': {'type': 'integer', 'minimum': 0}})
    stage = obj({'mode': {'enum': ['prepare', 'completion_verify', 'audit']},
                 'actual_exit_code': {'const': 0},
                 'owner': {'enum': ['/root/cpu_data_preparation', '/root/memory_diagnosis']},
                 'native_PID': {'type': 'integer', 'minimum': 1},
                 'parent_observed_wait': {'const': True},
                 'actual_exit_proof': binding,
                 'native_guard_closure': binding,
                 'result_receipt': binding,
                 'worker_source': binding})
    source = obj({'source_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                  'source_signature': signature,
                  'source_acquisition_receipt': binding,
                  'carrier_receipt': binding, 'payload': binding, 'row_index': binding,
                  'audit_receipt': binding,
                  'physical_rows': {'type': 'integer', 'minimum': 1},
                  'original_FLAC_bytes': {'type': 'integer', 'minimum': 1},
                  'original_preparation_outcome': obj({
                      'actual_exit_code': {'type': ['integer', 'null']},
                      'reason': {'type': 'string'},
                      'preserved_provenance': {'type': 'array', 'items': binding}}),
                  'preparation_or_completion_actual_closed': stage,
                  'independent_audit_actual_closed': stage})
    properties = {
        'kind': {'const': 'ASR18_original_FLAC_carrier_coverage_v1'},
        'schema_version': {'const': 1}, 'complete': {'const': True},
        'created_at_utc': {'type': 'string'}, 'owner': {'const': '/root/cpu_data_preparation'},
        'preparation_plan': binding, 'finalization_contract': binding,
        'completed_source_inventory': binding,
        'coverage_contract': binding,
        'scope': obj({'repository': {'const': 'serdarcaglar/kiraat'},
                      'revision': {'const': 'f8afe95abedc19ec6392c77cf470f5d3dcd2788f'},
                      'splits': {'const': ['train', 'rest']},
                      'expected_source_files': {'const': 729}}),
        'source_files': {'const': 729},
        'total_physical_rows': {'type': 'integer', 'minimum': 1},
        'total_original_FLAC_bytes': {'type': 'integer', 'minimum': 1},
        'total_cache_bytes': {'type': 'integer', 'minimum': 1, 'maximum': 400_000_000_000},
        'sources': {'type': 'array', 'minItems': 729, 'maxItems': 729, 'items': source},
        'all_physical_rows_and_original_compressed_bytes_independently_verified': {'const': True},
        'decoded_PCM_pool_created': {'const': False}, 'TRAIN_admitted': {'const': False}}
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema',
              '$id': 'ASR18_original_FLAC_carrier_coverage_v1', **obj(properties)}
    contract = {
        'kind': 'ASR18_compressed_carrier_coverage_contract_v1',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'final_receipt_path': str(AREA / 'KIRAAT_original_FLAC_carrier_coverage_v1.json'),
        'final_receipt_schema': schema,
        'cross_field_requirements': [
            'Every path must remain under designated E root except existing project worker-source bindings.',
            'All bindings require actual size/mtime/hash identities; finalizer uses previous full-byte audit hashes and frozen signatures, never claims a new bulk rehash.',
            '729 unique source hashes must equal all and only pinned train/rest inventory sources. No fixture/pilot duplicates or published dev/test.',
            'source_signature/source_acquisition_receipt must match completed source inventory and frozen source-identity inventory.',
            'Physical rows and original FLAC bytes must equal carrier receipt, independent all-row/all-byte audit, index count/contiguous physical row indices, and catalog physical-row references.',
            'preparation_or_completion_actual_closed.mode is prepare or completion_verify; independent_audit_actual_closed.mode is audit.',
            'Both stages have observed actual exit0 and distinct native children. Completion verification is read-only full-row/full-original-byte audit under a new owned child.',
            'Original preparation actual1 or unobserved outcome remains preserved. Payload complete never implies prep0. A fresh completion_verify0 plus separate audit0 may establish supported completion without relabeling old outcome.',
            'actual_exit_proof/native_guard_closure/result_receipt/worker_source identities must agree on source, mode, actual nativePID and actual observed wait; an inner Popen child need not have an invented tool session/chunk.',
            'Reuse fixture proofs only where parent actually observed wait. Rest00020 original dependency assertion prep1 is preserved and requires fresh completion_verify0 plus separate audit0.',
            'Sum physical rows/FLAC bytes equals outer totals; total cache bytes includes all cache files and stays <=400GB. All remaining fixed downloads plus E100GB reserve remain separate guard obligations.',
            'Final receipt absent until complete all729 source coverage. Partial milestones never masquerade as final coverage or TRAIN admission.'
        ],
        'no_source_files_or_labels_modified': True, 'no_PCM_or_transcoding': True,
        'full_coverage_currently_claimed': False, 'TRAIN_admitted': False}
    path = AREA / 'KIRAAT_FLAC_carrier_coverage_contract_v1.json'
    assert not path.exists()
    write_json(path, contract)
    print({'path': str(path), 'bytes': path.stat().st_size,
           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
