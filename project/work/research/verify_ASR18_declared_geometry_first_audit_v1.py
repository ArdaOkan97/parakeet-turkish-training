"""Independent metadata/mapping verification of the observed first new-audit exit0."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT, AREA
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
    guard = MetadataGuard('first_declared_geometry_binding_verifier', heavy=False, maximum_bytes=16_000_000)
    guard.wait()
    phase = AREA / 'KIRAAT_FLAC_carrier_batch08_v6'
    def pin(path):
        path = Path(path)
        before = path.stat()
        assert before.st_size < 8_000_000
        guard.pace(before.st_size)
        raw = path.read_bytes()
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
        return {'path': str(path.resolve()), 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(), 'mtime_ns': before.st_mtime_ns}
    def checked(value):
        actual = pin(value['path'])
        assert all(actual[key] == expected for key, expected in value.items())
        return read_json(value['path'])
    closed = read_json(phase / 'native_phase_CLOSED0.json')
    assert closed['native_return_code'] == 0 and closed['phase_source_files'] == 1
    completed = checked(closed['completed_sources'])
    assert len(completed) == 1
    item = completed[0]
    prepare = checked(item['prepared_actual_exit'])
    audit = checked(item['audited_actual_exit'])
    for proof, mode in ((prepare, 'prepare'), (audit, 'audit')):
        assert proof['actual_exit_code'] == 0 and proof['parent_observed_wait'] is True
        assert proof['mode'] == mode and proof['source_sha256'] == item['job']['sha256']
        footer = checked(proof['native_closure'])
        assert footer['PID'] == proof['native_PID'] and footer['native_return_code'] == 0
        assert footer['watchdog_clean'] is True and footer['peak_private_bytes'] < 2_000_000_000
    assert prepare['native_PID'] != audit['native_PID']
    result = read_json(Path(item['audited_actual_exit']['path']).parent / 'result.json')
    validation = checked(result['validation'])
    receipt = checked(validation['carrier_receipt'])
    assert validation['every_compressed_byte_and_physical_row_identical'] is True
    assert validation['full_physical_rows'] == receipt['physical_rows']
    assert validation['full_original_FLAC_bytes_compared'] == receipt['records']['bytes']
    assert validation['independent_full_payload_sha256'] == receipt['records']['sha256']
    geometry = validation['declared_STREAMINFO_geometry']
    assert geometry == result['declared_STREAMINFO_geometry']
    assert geometry['source_SHA256'] == item['job']['sha256']
    sidecar = geometry['sidecar']
    assert pin(sidecar['path']) == sidecar
    index = receipt['index']
    assert pin(index['path']) == index
    assert pin(geometry['parser']['path']) == geometry['parser']
    assert pin(geometry['writer']['path']) == geometry['writer']
    db = sqlite3.connect(Path(index['path']).as_uri()+'?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA cache_size=-1024')
    expected = db.execute('SELECT row_index,clip_id,source_id,sha256 FROM records ORDER BY row_index')
    guard.pace(sidecar['bytes'])
    count = unresolved = maximum_samples = maximum_frames = 0
    with Path(sidecar['path']).open('r', encoding='utf-8') as stream:
        header = json.loads(stream.readline())
        assert header['source_SHA256'] == item['job']['sha256']
        assert header['payload'] == receipt['records'] and header['row_index'] == index
        assert header['parser'] == geometry['parser'] and header['writer'] == geometry['writer']
        for line in stream:
            guard.wait()
            assert len(line) < 8192
            row = json.loads(line)
            wanted = expected.fetchone()
            assert wanted == (row['row_index'], row['clip_id'], row['source_id'], row['original_compressed_record_SHA256'])
            assert row['row_index'] == count
            value = row['geometry']
            assert len(value['STREAMINFO_prefix_SHA256']) == 64
            if row['status'].startswith('unresolved_'):
                unresolved += 1
            else:
                assert row['status'] == 'parsed_declared_geometry_runtime_match_required'
                assert value['sample_rate'] == 24000 and value['channels'] == 1
                samples = (value['total_native_samples']*16000+value['sample_rate']-1)//value['sample_rate']
                frames = (samples//160+7)//8
                assert value['clean16k_samples_from_header'] == samples
                assert value['mel_frames_from_header'] == samples//160
                assert value['encoder_frames_from_header'] == frames
            maximum_samples = max(maximum_samples, value.get('clean16k_samples_from_header') or 0)
            maximum_frames = max(maximum_frames, value.get('encoder_frames_from_header') or 0)
            count += 1
        assert expected.fetchone() is None
    db.close()
    assert count == geometry['physical_rows'] == receipt['physical_rows']
    assert unresolved == geometry['unresolved_declared_geometry_rows']
    assert maximum_samples == geometry['maximum_clean16k_samples_from_header']
    assert maximum_frames == geometry['maximum_encoder_frames_from_header']
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'transformers'))
    target = AREA / 'first_declared_geometry_audit_actual_v1.json'
    assert not target.exists()
    report = {'kind': 'ASR18_first_new_declared_geometry_audit_actual_binding_review_v1',
        'at_utc': now(), 'owner': '/root/cpu_data_preparation', 'PID': guard.process.pid, 'passed': True,
        'actual_owned_outer_session': 47877, 'actual_owned_outer_exit_code': 0,
        'actual_owned_outer_tool_chunk_id': '131d5f', 'native_phase_closure': pin(phase / 'native_phase_CLOSED0.json'),
        'prepared_actual_exit': item['prepared_actual_exit'], 'audited_actual_exit': item['audited_actual_exit'],
        'independent_source_all_byte_validation': result['validation'],
        'source_SHA256': item['job']['sha256'], 'physical_rows': count,
        'original_FLAC_bytes_independently_compared': receipt['records']['bytes'],
        'sidecar': sidecar, 'unresolved_declared_geometry_rows': unresolved,
        'maximum_clean16k_samples_from_header': maximum_samples,
        'maximum_encoder_frames_from_header': maximum_frames,
        'every_row_index_ID_record_SHA_mapping_and_sidecar_SHA_checked': True,
        'header_declaration_not_full_decoded_validation': True,
        'runtime_native_resampled_encoder_length_match_before_draw_required': True,
        'payload_audio_read_by_this_verifier': False, 'new_decoder_or_model_import': False,
        'private_bytes': guard.process.memory_info().private,
        'verifier_source': pin(__file__), 'full729_geometry_coverage': False, 'TRAIN_admitted': False}
    write_json(target, report)
    print({'report': str(target), 'rows': count, 'unresolved': unresolved,
           'maximum_encoder_T': maximum_frames, 'passed': True}, flush=True)


if __name__ == '__main__':
    main()
