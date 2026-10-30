"""Bounded cached label/ID alignment audit; never read audio or import models."""
import array
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import sys

from ASR18_data_io_guard_v2 import DataIOGuard
from cpu_preparation_guard_v2 import now, write_json

ROOT = Path(r'E:\gpu_training_scratchpad')
AREA = ROOT / 'analysis/BPE_weak_then_clean_v1'
PROJECT = Path(__file__).resolve().parents[2]
IO = DataIOGuard('label_alignment_audit', heavy=False, maximum_bytes=12_000_000)


def main():
    from tokenizers import Tokenizer
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    state = json.loads((AREA / 'BPE_label_preparation_status_v2.json').read_text())
    assert state['label_rows'] >= 839657 and state['frontier'] >= 839657
    candidate = Path(state['tokenizer_source'])
    assert IO.sha(candidate) == state['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(str(candidate))
    assert tokenizer.get_vocab_size() == 8711 and sys.byteorder == 'little'
    rng = random.Random(180217)
    groups = [('Alania', 1, 835251, 64),
              ('KIRAAT_rest00000', 835252, 837102, 32),
              ('KIRAAT_train00000', 837103, 839657, 32)]
    selected = [{'group': group, 'ordinal': item}
                for group, first, last, count in groups
                for item in sorted(rng.sample(range(first, last + 1), count))]
    manifest = {'seed': 180217, 'pinned_before_checks': True,
                'source_catalog_frontier': 839657, 'clips': selected,
                'quality_filters': [], 'audio_read': False}
    manifest_path = AREA / 'BPE_label_alignment_audit_manifest_v2.json'
    assert not manifest_path.exists(), 'Do not overwrite an existing pinned audit'
    write_json(manifest_path, manifest)
    source = sqlite3.connect((ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite').as_uri() + '?mode=ro', uri=True)
    labels = sqlite3.connect((ROOT / 'indexes/BPE_broad_labels_8711_v1.sqlite').as_uri() + '?mode=ro', uri=True)
    for db in (source, labels):
        db.execute('PRAGMA cache_size=-1024')
        db.execute('PRAGMA query_only=ON')
    ledger_path = AREA / 'BPE_label_alignment_audit_ledger_v2.jsonl'
    assert not ledger_path.exists()
    passed = 0
    with ledger_path.open('x', encoding='utf-8', buffering=1) as ledger:
        for selected_row in selected:
            IO.pace(65536)  # Bound random primary-key reads, no full table scan.
            ordinal = selected_row['ordinal']
            clip = source.execute('SELECT unique_id,dataset,original_split,text FROM clips WHERE ordinal=?', (ordinal,)).fetchone()
            label = labels.execute('SELECT target_ids_u32,target_tokens,text_sha256 FROM labels WHERE ordinal=?', (ordinal,)).fetchone()
            assert clip is not None and label is not None
            assert len(clip[3].encode()) + len(label[0]) < 32768
            packed = array.array('I'); packed.frombytes(label[0])
            recomputed = tokenizer.encode(clip[3], add_special_tokens=True).ids
            matches = (list(packed) == recomputed and len(packed) == label[1]
                       and hashlib.sha256(clip[3].encode()).hexdigest() == label[2]
                       and all(0 <= item < 8711 and item != 8192 for item in packed))
            row = {**selected_row, 'unique_id': clip[0], 'dataset': clip[1], 'original_split': clip[2],
                   'tokens': label[1], 'exact_tokenizer_text_and_blob_match': matches, 'at_utc': now()}
            ledger.write(json.dumps(row, ensure_ascii=False) + '\n'); ledger.flush()
            assert matches, 'Source/target alignment mismatch; preserve all source data'
            passed += 1
    source.close(); labels.close()
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    result = {'kind': 'ASR18_cached_BPE_alignment_audit_v2', 'PID': os.getpid(), 'at_utc': now(),
              'samples': len(selected), 'exact_matches': passed, 'actual_exit_code': 0,
              'seed': 180217, 'manifest_path': str(manifest_path), 'ledger_path': str(ledger_path),
              'tokenizer_sha256': state['tokenizer_sha256'], 'maximum_paced_IO_bytes': IO.maximum_bytes,
              'actual_paced_IO_bytes': IO.bytes, 'own_private_bytes': IO.process.memory_info().private,
              'audio_read': False, 'model_runtime_imported': False, 'quality_filters': [],
              'TRAIN_admitted': False, 'full_population_alignment_proven': False}
    write_json(AREA / 'BPE_label_alignment_audit_result_v2.json', result)
    IO.publish('completed_cached_label_alignment_audit', checked=passed)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
