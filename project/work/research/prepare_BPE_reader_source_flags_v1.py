"""Bound original source metadata to the already-pinned reader examples; no audio."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from ASR18_continuous_data_guard_v4 import DataIOGuard, ROOT, AREA
from cpu_preparation_guard_v2 import now, write_json


def main():
    guard = DataIOGuard('reader_original_flags_metadata', heavy=False, maximum_bytes=2_000_000)
    guard.wait()
    output = AREA / 'production_reader_contract_v1'
    target = output / 'original_source_flags.jsonl'
    assert not target.exists(), 'Keep the first sidecar immutable'
    manifest = output / 'manifest.jsonl'
    manifest_bytes = manifest.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.splitlines()]
    original = ROOT / 'indexes/alania_metadata_candidates_v1.sqlite'
    connection = sqlite3.connect(original.as_uri() + '?mode=ro&immutable=1', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA cache_size=-512')
    connection.execute('PRAGMA temp_store=MEMORY')
    catalog = sqlite3.connect((ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite').as_uri() + '?mode=ro', uri=True, timeout=2)
    catalog.execute('PRAGMA cache_size=-512')
    digest = catalog.execute("SELECT source_sha256 FROM files WHERE file='alania_all_metadata'").fetchone()[0]
    records = []
    for row in rows:
        guard.pace(65536)
        deadline = time.monotonic() + .25
        while time.monotonic() < deadline:
            guard.wait(); time.sleep(.025)
        if row['dataset'] == 'alania_yodas3':
            segment = row['unique_id'].removeprefix('alania:')
            result = dict(connection.execute('SELECT * FROM candidates WHERE segment_id=?', (segment,)).fetchone())
            assert result['recording_id'] == row['source_id']
            assert result['transcript'] == row['text']
            assert result['start_s'] == row['start'] and result['end_s'] == row['end']
            records.append({'ordinal': row['ordinal'], 'unique_id': row['unique_id'],
                            'source_metadata_path': str(original),
                            'source_metadata_SHA256_bound_by_catalog': digest,
                            'original_source_metadata': result,
                            'all_quality_language_disagreement_flags_ignored': True})
        else:
            records.append({'ordinal': row['ordinal'], 'unique_id': row['unique_id'],
                            'raw_flags_already_in_pinned_manifest': row['raw_flags'],
                            'all_recommended_quality_language_flags_ignored': True})
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode('utf-8')
    target.write_bytes(encoded)
    write_json(output / 'original_source_flags_receipt.json', {
        'at_utc': now(), 'PID': guard.process.pid, 'sidecar_path': str(target),
        'sidecar_SHA256': hashlib.sha256(encoded).hexdigest(),
        'manifest_SHA256': hashlib.sha256(manifest_bytes).hexdigest(), 'records': len(records),
        'actual_paced_metadata_bytes': guard.bytes,
        'private_bytes': guard.process.memory_info().private,
        'source_metadata_preserved_in_place': True, 'source_flags_used_for_selection': False,
        'new_audio_read_or_decode': False, 'TRAIN_admitted': False})
    print(json.dumps({'PID': guard.process.pid, 'sidecar_path': str(target),
                      'sidecar_SHA256': hashlib.sha256(encoded).hexdigest(), 'records': len(records)}), flush=True)
    connection.close(); catalog.close()


if __name__ == '__main__':
    main()
