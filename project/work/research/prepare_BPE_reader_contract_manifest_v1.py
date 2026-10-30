"""Bounded metadata-only production-reader examples; no audio or GPU imports."""
import array
import hashlib
import heapq
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import time

from ASR18_continuous_data_guard_v4 import DataIOGuard, ROOT, AREA
from cpu_preparation_guard_v2 import now, write_json

PROJECT = Path(__file__).resolve().parents[2]
OUT = AREA / 'production_reader_contract_v1'
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from metrics import normalize_text


class MetadataGuard(DataIOGuard):
    def pace(self, count):
        super().pace(count)
        deadline = max(getattr(self, 'metadata_next', 0), time.monotonic()) + count / 262144
        while time.monotonic() < deadline:
            self.wait(); time.sleep(min(.025, max(0, deadline - time.monotonic())))
        self.metadata_next = deadline


def ro(path):
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=2)
    db.execute('PRAGMA cache_size=-1024'); db.execute('PRAGMA temp_store=MEMORY')
    db.execute('PRAGMA query_only=ON'); return db


def main():
    io = MetadataGuard('reader_contract_metadata', heavy=False, maximum_bytes=128_000_000)
    io.state['maximum_additional_metadata_IO_bytes_s'] = 262144
    io.wait(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'manifest.jsonl').exists(), 'Keep the first pinned manifest immutable'
    plan = json.loads((AREA / 'preparation_plan_v1.json').read_text())
    bindings = json.loads((AREA / 'heldout_metadata_bindings_v1.json').read_text())
    heldout_sources = set(bindings['source_ids']); heldout_text = set(bindings['normalized_text_keys'])
    source_path = ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite'
    labels_path = ROOT / 'indexes/BPE_broad_labels_8711_v1.sqlite'
    source = ro(source_path); labels = ro(labels_path)
    source.row_factory = sqlite3.Row; labels.row_factory = sqlite3.Row
    frontier = min(source.execute('SELECT MAX(ordinal) FROM clips').fetchone()[0],
                   labels.execute('SELECT frontier FROM progress WHERE id=1').fetchone()[0])
    alania_rows = source.execute("SELECT rows_retained FROM files WHERE file='alania_all_metadata'").fetchone()[0]
    ranges = [('alania_yodas3', 1, min(8192, alania_rows)), ('kiraat', alania_rows + 1, frontier)]
    scan_plan = {'pinned_before_scan': True, 'ordinal_ranges_inclusive': ranges, 'seed': 180317,
                 'largest_means_only_within_scanned_ranges': True, 'global_corpus_complete': False,
                 'maximum_paced_metadata_bytes': 128_000_000, 'maximum_metadata_IO_bytes_s': 262144,
                 'selection_is_reader_test_coverage_not_TRAIN_filter': True, 'audio_read': False}
    write_json(OUT / 'pinned_scan_plan.json', scan_plan)
    largest = {name: [] for name, _, _ in ranges}; reservoirs = {}; scanned = 0
    for dataset, first, last in ranges:
        for start in range(first, last + 1, 512):
            stop = min(start + 511, last); io.pace((stop - start + 1) * 512)
            rows = labels.execute('SELECT ordinal,target_tokens,conservative_estimated_joint_bytes FROM labels '
                                  'WHERE ordinal BETWEEN ? AND ? ORDER BY ordinal', (start, stop)).fetchall()
            for row in rows:
                ordinal, tokens, dense = tuple(row); scanned += 1
                heapq.heappush(largest[dataset], (dense, ordinal))
                if len(largest[dataset]) > (2 if dataset == 'alania_yodas3' else 4): heapq.heappop(largest[dataset])
                group = 'small' if dense <= 64 * 1024**2 else 'moderate' if dense <= 512 * 1024**2 else None
                if group:
                    ranking = hashlib.sha256(f'ASR18reader180317:{dataset}:{group}:{ordinal}'.encode()).hexdigest()
                    sample = reservoirs.setdefault((dataset, group), []); sample.append((ranking, ordinal))
                    sample.sort(); del sample[2:]
            io.publish('scanning_available_geometry_metadata', scanned_label_rows=scanned,
                       scan_frontier=stop, maximum_additional_metadata_IO_bytes_s=262144)
    selected = {}
    for (dataset, group), choices in reservoirs.items():
        for _, ordinal in choices: selected.setdefault(ordinal, []).append(dataset + ':' + group)
    for dataset, values in largest.items():
        for _, ordinal in sorted(values, reverse=True): selected.setdefault(ordinal, []).append(dataset + ':largest_in_scanned_range')
    jobs = plan['download_jobs'] + plan['reused_archives']
    yodas = {Path(j['path']).stem: j for j in jobs if j['repository'] == 'espnet/yodas3'}
    kiraat = {str(Path(j['destination'])).casefold(): j for j in jobs if j['repository'] == 'serdarcaglar/kiraat'}
    from tokenizers import Tokenizer
    token_state = json.loads((AREA / 'BPE_label_preparation_status_v2.json').read_text())
    tokenizer_path = Path(token_state['tokenizer_source'])
    assert io.sha(tokenizer_path) == token_state['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); assert tokenizer.get_vocab_size() == 8711
    schemas = {'catalog': [dict(x) for x in source.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")],
               'BPE': [dict(x) for x in labels.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")]}
    records = []
    for ordinal in sorted(selected):
        io.pace(65536)
        clip = dict(source.execute('SELECT * FROM clips WHERE ordinal=?', (ordinal,)).fetchone())
        label = dict(labels.execute('SELECT * FROM labels WHERE ordinal=?', (ordinal,)).fetchone())
        ids = array.array('I'); ids.frombytes(label.pop('target_ids_u32'))
        assert sys.byteorder == 'little' and len(ids) == label['target_tokens']
        assert all(0 <= value < 8711 and value != 8192 for value in ids)
        assert hashlib.sha256(clip['text'].encode()).hexdigest() == label['text_sha256']
        assert tokenizer.encode(clip['text'], add_special_tokens=True).ids == list(ids)
        assert clip['source_id'] not in heldout_sources
        assert not any(normalize_text(clip[name], 'tr') in heldout_text for name in ('text', 'text_raw', 'text_spoken') if clip[name])
        flags = json.loads(clip['original_flags_json'])
        if clip['dataset'] == 'alania_yodas3':
            shard = Path(clip['shard']).stem
            if shard.isdigit(): shard = shard.zfill(4)
            job = yodas[shard]; index_path = ROOT / f'indexes/yodas3_tar_{shard}.sqlite'
            locator = ro(index_path); locator.row_factory = sqlite3.Row
            member = locator.execute('SELECT * FROM recordings WHERE recording_id=?', (clip['source_id'],)).fetchone()
            assert member is not None; member = dict(member)
            schemas.setdefault('TAR_index', [dict(x) for x in locator.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")]); locator.close()
            name = PurePosixPath(member['member_name'])
            assert not name.is_absolute() and '..' not in name.parts
            assert 0 <= member['offset'] < member['offset'] + member['bytes'] <= job['size']
            lookup = {'kind': 'original_Opus_WebM_TAR_member', 'index_path': str(index_path), **member,
                      'reader_interval_seconds': [clip['start'], clip['end']],
                      'PTS_origin_requires_reader_validation': True, 'do_not_apply_old_20second_or_eligible_gate': True}
        else:
            job = kiraat[str(Path(clip['shard'])).casefold()]
            lookup = {'kind': 'embedded_clip_FLAC_Parquet_row', 'row_index': clip['row_index'],
                      'source_recording_interval_seconds': [clip['start'], clip['end']],
                      'decode_complete_embedded_clip_without_second_source_trim': True,
                      'source_card_channels': 1, 'source_card_sample_rate_Hz': 24000,
                      'full_audio_column_rowgroup_materialization_forbidden': True}
        path = Path(job['destination']); receipt_path = path.with_suffix(path.suffix + '.receipt.json')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        assert path.exists() and path.stat().st_size == job['size']
        assert receipt['sha256'] == job['sha256'] and receipt['revision'] == job['revision']
        lookup.update(source_path=str(path), repository=job['repository'], source_revision=job['revision'],
                      source_size_bytes=job['size'], source_SHA256=job['sha256'],
                      acquisition_receipt_path=str(receipt_path), acquisition_receipt_SHA256=io.sha(receipt_path))
        records.append({**clip, 'raw_flags': flags, 'validation_coverage': selected[ordinal], 'audio_lookup': lookup,
                        'BPE': {**label, 'target_ids': list(ids), 'vocabulary': 8711, 'blank_id': 8192,
                                'TDT_duration_rows_not_in_targets': [8711, 8716]},
                        'metadata_dense_joint_estimate_is_not_measured_native_memory': True,
                        'nominal_16k_samples_for_planning_only': round(clip['duration'] * 16000),
                        'actual_clean_samplecount_and_encoder_frames_require_decode': True,
                        'source_and_normalized_reference_exclusion_checked': True,
                        'complete_audio_heldout_exclusion_claimed': False, 'TRAIN_admitted': False})
    manifest = OUT / 'manifest.jsonl'
    with manifest.open('x', encoding='utf-8') as stream:
        for row in records: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    heldout_path = AREA / 'heldout_metadata_bindings_v1.json'
    contract = {'kind': 'ASR18_metadata_only_unfiltered_reader_contract_v1', 'at_utc': now(),
        'PID': io.process.pid, 'manifest_path': str(manifest), 'manifest_SHA256': io.sha(manifest),
        'records': len(records), 'scanned_label_rows': scanned, 'scanned_ranges': ranges, 'schemas': schemas,
        'catalog_path': str(source_path), 'BPE_path': str(labels_path), 'source_frontier_snapshot': frontier,
        'tokenizer_path': str(tokenizer_path), 'tokenizer_SHA256': token_state['tokenizer_sha256'],
        'normalization_source': str(PROJECT / 'outputs/parakeet_turkish/metrics.py'),
        'normalization_source_SHA256': io.sha(PROJECT / 'outputs/parakeet_turkish/metrics.py'),
        'label_contract': 'Encode supplied orthographic text unchanged, add_special_tokens=True; raw/spoken variants and all flags retained',
        'normalization_purpose': 'Heldout identity/reference exclusion only; not a target rewrite or quality filter',
        'heldout_binding_path': str(heldout_path), 'heldout_binding_SHA256': io.sha(heldout_path),
        'heldout_source_ids': len(heldout_sources), 'heldout_normalized_text_keys': len(heldout_text),
        'heldout_source_manifest_bindings': bindings['files'], 'TEST_predictions_read': False,
        'audio_heldout_exclusion_still_required': True, 'audio_read_or_decoded': False,
        'clean_reader_output_contract': 'Deterministic mono float32 16000 Hz before augmentation; report actual sample count; no duration/quality/LID selection',
        'geometry_contract': 'Stored dense estimate uses (ceil(duration*12.5)+2)*(U+1)*8716*4; actual decoded sample count and native encoder frame count must be measured',
        'full_corpus_finalization': {'new_download_files': 773, 'reused_archives': 11, 'expected_metadata_files': 730,
            'expected_YODAS_header_indexes': 51,
            'prepared_source_receipt': str(AREA / 'continuous_finalization_successors_prepared_v3.json'),
            'catalog_marker': str(AREA / 'broad_metadata_actual_CLOSED0_v3.json'),
            'BPE_marker': str(AREA / 'BPE_labels_actual_CLOSED0_v3.json'),
            'require_actual_owned_exit_receipts_and_whole_DB_SHA256': True,
            'require_matching_catalog_rows_label_rows_and_frontier': True,
            'weak_epoch_denominator': 'Final retained catalog row count; each of two weak passes must visit each retained ordinal exactly once. Retention draws have a separate clock.',
            'mean_U': 'Final total target tokens divided by final label rows, not this partial mean or a sampled estimate',
            'clean_phase': 'Separate fixed352768-clip/675.0964h corpus; bind its own final targets and denominator',
            'full_acquisition_and_reader_geometry_and_heldout_admission_are_separate': True},
        'complete_corpus': False, 'TRAIN_admitted': False, 'all_current_workers_unchanged': True,
        'no_Torch_or_audio_imports': 'torch' not in sys.modules and 'av' not in sys.modules,
        'actual_paced_metadata_bytes': io.bytes, 'own_private_bytes': io.process.memory_info().private}
    source.close(); labels.close(); write_json(OUT / 'contract.json', contract)
    io.publish('completed_metadata_only_reader_manifest', selected_records=len(records), scanned_label_rows=scanned)
    print(json.dumps({k: contract[k] for k in ('PID', 'manifest_path', 'manifest_SHA256', 'records', 'scanned_label_rows', 'own_private_bytes', 'actual_paced_metadata_bytes')}))


if __name__ == '__main__':
    main()
