"""Bounded header-only fixture against already measured canonical CPU decodes."""
import hashlib
from pathlib import Path
import sqlite3
import sys
import time

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT, AREA
from ASR18_FLAC_streaminfo_geometry_v1 import streaminfo_geometry, SPECIFICATION
from cpu_preparation_guard_v2 import now

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json


class MetadataGuard(DataIOGuard):
    def pace(self, count):
        super().pace(count)
        end = time.monotonic()+count/262144
        while time.monotonic() < end:
            self.wait()
            time.sleep(min(.025, max(0, end-time.monotonic())))


def main():
    guard = MetadataGuard('STREAMINFO_header_fixture', heavy=False, maximum_bytes=8_000_000)
    guard.wait()
    output = AREA / 'FLAC_STREAMINFO_geometry_header_fixture_v1'
    output.mkdir(exist_ok=False)

    def binding(path):
        path = Path(path)
        assert path.stat().st_size < 8_000_000
        guard.pace(path.stat().st_size)
        return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    reference_dir = ROOT / 'analysis/ASR18_memory_diagnosis_v1'
    first = reference_dir / 'broad_audio_actual_01/report.json'
    full = reference_dir / 'full14_FLAC_CPU_actual_01/decoded/report.json'
    hard = reference_dir / 'hard_U_rows_decode_01/decoded/report.json'
    reference_pins = [binding(first), binding(full), binding(hard)]
    references = [value for value in read_json(first)['observations']
                  if value['format'] == 'KIRAAT_embedded_FLAC']
    references += [value['reader_evidence'] for value in read_json(full)['rows']
                   if value['row']['dataset'] == 'kiraat']
    references += [value['reader_evidence'] for value in read_json(hard)['rows']]
    assert len(references) >= 6
    observations = []
    last_prefix = None
    for reference in references:
        guard.wait()
        sha = reference['source_file_sha256']
        folder = ROOT / 'cache/KIRAAT_original_FLAC_records_v1' / sha
        receipt_path = folder / 'receipt.json'
        receipt = read_json(receipt_path)
        assert receipt['complete'] is True and receipt['source']['source_sha256'] == sha
        index = receipt['index']
        assert binding(index['path']) == {key: index[key] for key in ('path', 'bytes', 'sha256')}
        db = sqlite3.connect(Path(index['path']).as_uri()+'?mode=ro&immutable=1', uri=True)
        db.execute('PRAGMA cache_size=-1024')
        row = db.execute('SELECT clip_id,source_id,offset,bytes,sha256 FROM records '
                         'WHERE row_index=?', (reference['original_row_index'],)).fetchone()
        db.close()
        assert row is not None and 'kiraat:'+row[0] == reference['unique_id']
        payload = receipt['records']
        path = Path(payload['path'])
        before = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (payload['bytes'], payload['mtime_ns'])
        assert 42 <= row[3] <= 32*1024**2
        guard.pace(42)
        with path.open('rb', buffering=0) as stream:
            stream.seek(row[2])
            prefix = stream.read(42)
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
        geometry = streaminfo_geometry(prefix)
        assert geometry['known_total_samples'] is True
        assert geometry['sample_rate'] == reference['native_sample_rate']
        assert geometry['total_native_samples'] == reference['native_samples']
        assert geometry['clean16k_samples_from_header'] == reference['decoded_samples']
        assert geometry['encoder_frames_from_header'] == reference['encoder_frames']
        observations.append({'unique_id': reference['unique_id'], 'source_SHA256': sha,
            'physical_row_index': reference['original_row_index'], 'prior_audited_record_SHA256': row[4],
            'carrier_receipt': binding(receipt_path), 'geometry': geometry,
            'matches_existing_measured_native_and_clean16k_and_encoder_lengths': True})
        last_prefix = prefix
    assert last_prefix is not None
    unknown = bytearray(last_prefix)
    packed = int.from_bytes(unknown[18:26], 'big') & ~((1 << 36)-1)
    unknown[18:26] = packed.to_bytes(8, 'big')
    value = streaminfo_geometry(unknown)
    assert value['known_total_samples'] is False and value['clean16k_samples_from_header'] is None
    rejected = []
    for name, invalid in (('truncated_prefix', last_prefix[:41]),
                          ('wrong_marker', b'xxxx'+last_prefix[4:]),
                          ('wrong_first_metadata_type', last_prefix[:4]+b'\x81'+last_prefix[5:])):
        try:
            streaminfo_geometry(invalid)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Malformed header was not rejected: '+name)
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'transformers'))
    report = {'kind': 'ASR18_FLAC_STREAMINFO_geometry_header_fixture_v1', 'at_utc': now(),
        'owner': '/root/cpu_data_preparation', 'PID': guard.process.pid, 'passed': True,
        'parser': binding(PROJECT / 'work/research/ASR18_FLAC_streaminfo_geometry_v1.py'),
        'fixture_source': binding(__file__), 'specification': SPECIFICATION,
        'existing_canonical_CPU_decode_reference_reports': reference_pins,
        'observations': observations, 'new_compressed_prefix_bytes_read': 42*len(observations),
        'unknown_total_samples_remains_unresolved': True, 'malformed_controls_rejected': rejected,
        'new_audio_decode_or_model_import': False, 'existing_cache_or_labels_modified': False,
        'all_corpus_decoded_geometry_certified': False, 'TRAIN_admitted': False,
        'actual_paced_metadata_bytes': guard.bytes, 'private_bytes': guard.process.memory_info().private}
    write_json(output / 'report.json', report)
    print({'report': str(output / 'report.json'), 'actual_header_examples': len(observations),
           'new_header_bytes_read': report['new_compressed_prefix_bytes_read'],
           'private_bytes': report['private_bytes'], 'passed': True}, flush=True)


if __name__ == '__main__':
    main()
