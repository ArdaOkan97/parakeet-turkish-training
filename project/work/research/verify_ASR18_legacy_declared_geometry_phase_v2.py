"""Independent complete metadata review for one finite legacy-header phase."""
import argparse
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import (
    MetadataIO, actual_parent, actual_child, sidecar_rows)
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual-phase', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    target = Path(args.report).resolve()
    assert target.is_relative_to(ROOT.resolve()) and not target.exists()
    guard = DataIOGuard('legacy_declared_geometry_phase_independent_review', heavy=False,
                        maximum_bytes=400_000_000)
    io = MetadataIO(guard)
    closed = actual_parent(io, args.actual_phase, legacy=True)
    finished = io.checked(closed['completed_sources'])
    assert 1 <= len(finished) <= 32
    rows = bytes_verified = unresolved = 0
    sidecars = []
    for item in finished:
        result, footer = actual_child(io, item['geometry_completion_actual_exit'],
                                      item['job']['sha256'], 'declared_geometry_completion')
        assert result == item['geometry_result']
        assert result['all_record_SHA_and_payload_SHA_verified'] is True
        geometry = result['declared_STREAMINFO_geometry']
        sidecar = geometry['sidecar']
        guard.pace(64 * 1024)
        with Path(sidecar['path']).open('r', encoding='utf-8') as stream:
            import json
            header = json.loads(stream.readline(64 * 1024))
        receipt = io.checked(header['carrier_receipt'])
        entry = dict(source_SHA256=item['job']['sha256'],
            geometry_sidecar=sidecar, row_index=receipt['index'],
            carrier_receipt=io.binding(header['carrier_receipt']['path']), payload=receipt['records'],
            geometry_parser=geometry['parser'], geometry_writer=geometry['writer'],
            physical_rows=geometry['physical_rows'], unresolved_physical_rows=geometry['unresolved_declared_geometry_rows'])
        mapping, counts = sidecar_rows(io, entry)
        rows += len(mapping)
        bytes_verified += result['existing_FLAC_payload_bytes_read_and_SHA_verified']
        unresolved += sum(counts.values())
        sidecars.append(dict(source_SHA256=item['job']['sha256'], physical_rows=len(mapping),
            sidecar=sidecar, unresolved_rows=sum(counts.values())))
    assert rows == closed['physical_rows'] and unresolved == closed['unresolved_declared_geometry_rows']
    assert bytes_verified == closed['all_record_and_payload_SHA_verified_bytes']
    io.unchanged()
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    value = dict(kind='ASR18_finite_legacy_declared_geometry_independent_mapping_review_v1',
        owner='/root/cpu_data_preparation', at_utc=now(), PID=guard.process.pid, passed=True,
        actual_phase=io.binding(args.actual_phase), source_count=len(finished), physical_rows=rows,
        previously_worker_verified_original_FLAC_bytes=bytes_verified,
        metadata_verifier_audio_payload_bytes_read=0, unresolved_declared_geometry_rows=unresolved,
        every_sidecar_SHA_row_index_clipID_sourceID_recordSHA_formula_checked=True,
        sources=sidecars, verifier_source=io.binding(__file__),
        producer_metadata_helpers=io.binding(Path(__file__).with_name('finalize_ASR18_declared_training_geometry_v1.py')),
        own_private_bytes=guard.process.memory_info().private,
        full729_geometry_complete=False, TRAIN_admitted=False)
    write_json(target, value)
    print(dict(report=io.binding(target), source_count=len(finished), physical_rows=rows,
        unresolved_rows=unresolved, TRAIN_admitted=False), flush=True)


if __name__ == '__main__':
    main()
