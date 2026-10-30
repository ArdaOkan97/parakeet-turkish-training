"""One readonly old-carrier payload pass: every record hash plus declared headers."""
import argparse
import hashlib
import os
from pathlib import Path
import sqlite3
import sys

from ASR18_FLAC_carrier_guard_v6 import CarrierGuard
from ASR18_FLAC_carrier_guard_v3 import CACHE, metadata, signature
from ASR18_FLAC_geometry_sidecar_v1 import GeometrySidecar
from prepare_ASR18_KIRAAT_FLAC_records_v8 import verify_source
from cpu_preparation_guard_v2 import now
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json


def complete(config, guard):
    source = verify_source(config, guard)
    for key in ('original_prepared_actual_exit', 'original_audited_actual_exit'):
        pin = config[key]
        assert guard.binding(pin['path']) == pin
        old = metadata(pin['path'])
        assert old['actual_exit_code'] == 0 and old['parent_observed_wait'] is True
        assert old['source_sha256'] == config['job']['sha256']
    folder = CACHE / config['job']['sha256']
    receipt_path = folder / 'receipt.json'
    receipt_pin = guard.binding(receipt_path)
    receipt = metadata(receipt_path)
    assert receipt['complete'] is True and receipt['source']['source_sha256'] == config['job']['sha256']
    payload, index = receipt['records'], receipt['index']
    assert signature(payload['path']) == {key: payload[key] for key in ('path', 'bytes', 'mtime_ns')}
    assert guard.binding(index['path']) == {key: index[key] for key in ('path', 'bytes', 'sha256')}
    db = sqlite3.connect(Path(index['path']).as_uri()+'?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA cache_size=-1024')
    db.execute('PRAGMA temp_store=MEMORY')
    ledger = GeometrySidecar(config, guard, receipt_path, receipt)
    guard.flush_hook = ledger.flush
    count = total = 0
    digest = hashlib.sha256()
    ordered = hashlib.sha256()
    import json
    try:
        with Path(payload['path']).open('rb', buffering=0) as stream:
            for ordinal, clip, source_id, offset, size, sha in db.execute(
                'SELECT row_index,clip_id,source_id,offset,bytes,sha256 FROM records ORDER BY row_index'):
                guard.wait()
                assert ordinal == count and offset == total and 0 < size <= 32*1024**2
                actual = guard.read(stream, size)
                assert len(actual) == size and hashlib.sha256(actual).hexdigest() == sha
                row = (clip, source_id, offset, size, sha)
                digest.update(actual)
                ordered.update(json.dumps([count, *row], separators=(',', ':')).encode())
                ledger.add(count, row, actual)
                count += 1
                total += size
                del actual
            assert not stream.read(1)
        assert count == receipt['physical_rows']
        assert total == payload['bytes'] and digest.hexdigest() == payload['sha256']
        assert ordered.hexdigest() == receipt['ordered_mapping_sha256']
    finally:
        guard.flush_hook = lambda: None
        ledger.close_partial()
        db.close()
    assert signature(payload['path']) == {key: payload[key] for key in ('path', 'bytes', 'mtime_ns')}
    assert signature(index['path']) == {key: index[key] for key in ('path', 'bytes', 'mtime_ns')}
    assert guard.binding(receipt_path) == receipt_pin
    verify_source(config, guard)
    geometry = ledger.finish(count)
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    return {'kind': 'ASR18_readonly_old_carrier_declared_geometry_completion_v1',
        'source_SHA256': config['job']['sha256'], 'physical_rows': count,
        'existing_FLAC_payload_bytes_read_and_SHA_verified': total,
        'all_record_SHA_and_payload_SHA_verified': True,
        'original_prepared_actual_exit': config['original_prepared_actual_exit'],
        'original_audited_actual_exit': config['original_audited_actual_exit'],
        'declared_STREAMINFO_geometry': geometry, 'original_source_and_carrier_files_modified': False,
        'new_Parquet_or_audio_decoder_read': False, 'TRAIN_admitted': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = metadata(args.config)
    guard = CarrierGuard(config)
    code = 0
    try:
        result = complete(config, guard)
        write_json(guard.run / 'result.json', {**result, 'mode': config['mode'],
            'PID': os.getpid(), 'at_utc': now(), 'actual_paced_IO_bytes': guard.bytes,
            'actual_new_payload_bytes': guard.payload_written})
        print({'source': config['job']['path'], 'physical_rows': result['physical_rows'],
               'actual_paced_IO_bytes': guard.bytes}, flush=True)
    except BaseException as error:
        code = 75 if type(error).__name__ == 'DataStop' else 1
        write_json(guard.run / 'failure.json', {'at_utc': now(), 'error': type(error).__name__,
            'reason': str(error), 'originals_and_partial_sidecars_preserved': True})
        raise
    finally:
        guard.close(code)


if __name__ == '__main__':
    main()
