"""Read-only geometry envelope of a pinned committed BPE frontier; no audio."""
import hashlib
import heapq
import json
import math
from pathlib import Path
import sqlite3
import time

from ASR18_continuous_data_guard_v4 import DataIOGuard, ROOT, AREA
from cpu_preparation_guard_v2 import now, write_json


class MetadataGuard(DataIOGuard):
    def pace(self, count):
        super().pace(count)
        end = time.monotonic() + count / 262144
        while time.monotonic() < end:
            self.wait(); time.sleep(min(.025, max(0, end-time.monotonic())))


def reader(path):
    result = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=2)
    result.execute('PRAGMA cache_size=-1024')
    result.execute('PRAGMA temp_store=MEMORY')
    result.execute('PRAGMA query_only=ON')
    return result


def main():
    guard = MetadataGuard('partial_geometry_metadata', heavy=False, maximum_bytes=128_000_000)
    guard.wait()
    out = AREA / 'partial_geometry_envelope_v1'
    out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists(), 'Preserve the first pinned envelope'
    source_path = ROOT/'indexes/BPE_broad_weak_labels_v1.sqlite'
    label_path = ROOT/'indexes/BPE_broad_labels_8711_v1.sqlite'
    source = reader(source_path); labels = reader(label_path)
    frontier, rows, tokens, review = labels.execute('SELECT frontier,rows,tokens,large_geometry_review_rows FROM progress WHERE id=1').fetchone()
    source_max = source.execute('SELECT MAX(ordinal) FROM clips').fetchone()[0]
    alania_raw, alania_rows = source.execute("SELECT rows_seen,rows_retained FROM files WHERE file='alania_all_metadata'").fetchone()
    metadata_files = source.execute('SELECT COUNT(*) FROM files').fetchone()[0]
    assert alania_raw == 835267 and frontier <= source_max and frontier >= alania_rows
    plan = {'created_at_utc': now(), 'PID': guard.process.pid, 'catalog_path': str(source_path),
            'label_path': str(label_path), 'pinned_label_frontier': frontier,
            'pinned_progress_rows': rows, 'pinned_progress_tokens': tokens,
            'source_frontier_at_start': source_max, 'metadata_files_at_start': metadata_files,
            'alania_retained_ordinal_end': alania_rows, 'KIRAAT_partial': metadata_files < 730,
            'scalar_metadata_read_charge_bytes_per_label': 32, 'metadata_rate_bytes_s': 262144,
            'maximum_metadata_read_budget': 128_000_000, 'audio_read': False, 'GPU_used': False,
            'source_and_labels_remain_unmodified': True, 'selection_exclusion_performed': False}
    write_json(out/'pinned_scan_plan.json', plan)
    groups = {key: {'rows': 0, 'sum_U': 0, 'MAX_U': 0, 'count_U_gt_458': 0,
                    'count_U_ge_1024': 0, 'count_U_plus_1_gt_1024': 0,
                    'MAX_estimated_T_times_U_plus_1': 0}
              for key in ('all', 'alania_yodas3', 'kiraat')}
    heaps = {(dataset, kind): [] for dataset in groups for kind in ('U', 'T_times_U_plus_1')}
    denominator = 8716 * 4
    for begin in range(1, frontier+1, 4096):
        end = min(frontier, begin+4095)
        guard.pace((end-begin+1)*32)
        batch = labels.execute('SELECT ordinal,target_tokens,conservative_estimated_joint_bytes '
                               'FROM labels WHERE ordinal BETWEEN ? AND ? ORDER BY ordinal', (begin,end)).fetchall()
        assert len(batch) == end-begin+1, 'A gap in the pinned label frontier must fail closed'
        for ordinal, U, dense in batch:
            assert dense % denominator == 0
            product = dense // denominator
            dataset = 'alania_yodas3' if ordinal <= alania_rows else 'kiraat'
            for group in ('all', dataset):
                g = groups[group]; g['rows'] += 1; g['sum_U'] += U
                g['MAX_U'] = max(g['MAX_U'], U)
                g['count_U_gt_458'] += U > 458; g['count_U_ge_1024'] += U >= 1024
                g['count_U_plus_1_gt_1024'] += U+1 > 1024
                g['MAX_estimated_T_times_U_plus_1'] = max(g['MAX_estimated_T_times_U_plus_1'], product)
                for kind, value in (('U',U), ('T_times_U_plus_1',product)):
                    heap = heaps[(group,kind)]
                    heapq.heappush(heap,(value,ordinal))
                    if len(heap)>8: heapq.heappop(heap)
        guard.publish('scanning_pinned_committed_geometry', scan_frontier=end, pinned_frontier=frontier,
                      rows_scanned=groups['all']['rows'], maximum_additional_metadata_IO_bytes_s=262144)
    assert groups['all']['rows'] == rows == frontier
    assert groups['all']['sum_U'] == tokens
    chosen = {}
    for (dataset, kind), heap in heaps.items():
        for _, ordinal in sorted(heap,reverse=True): chosen.setdefault(ordinal,[]).append(dataset+':top_'+kind)
    source.row_factory = sqlite3.Row
    examples = []
    for ordinal, selected_by in sorted(chosen.items()):
        guard.pace(16384)
        raw = dict(source.execute('SELECT * FROM clips WHERE ordinal=?',(ordinal,)).fetchone())
        U, dense = labels.execute('SELECT target_tokens,conservative_estimated_joint_bytes FROM labels WHERE ordinal=?',(ordinal,)).fetchone()
        T = math.ceil(raw['duration']*12.5)+2
        assert T*(U+1)*denominator == dense
        examples.append({**raw, 'U':U, 'estimated_T':T, 'estimated_T_times_U_plus_1':T*(U+1),
                         'estimated_dense_joint_bytes':dense, 'selected_by':selected_by,
                         'audio_lookup_already_in_catalog':True, 'actual_encoder_frames_measured':False,
                         'rows_remain_retained':True, 'TRAIN_admitted':False})
    encoded = ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in examples).encode()
    (out/'top_rows.jsonl').write_bytes(encoded)
    report = {**plan, 'completed_at_utc':now(), 'groups':groups,
              'top_row_count':len(examples), 'top_rows_path':str(out/'top_rows.jsonl'),
              'top_rows_SHA256':hashlib.sha256(encoded).hexdigest(),
              'top_8_ordinals_by_group':{group:{kind:[ordinal for _,ordinal in sorted(heaps[(group,kind)],reverse=True)]
                                               for kind in ('U','T_times_U_plus_1')} for group in groups},
              'actual_paced_metadata_bytes':guard.bytes, 'private_bytes':guard.process.memory_info().private,
              'estimated_T_formula':'ceil(source duration seconds *12.5)+2; native encoder frames unmeasured',
              'geometry_estimate_is_not_actual_native_memory':True,
              'all_Alania_retained_metadata_and_labels_included':True,
              'all_KIRAAT_or_full_corpus_coverage_claimed':False,
              'no_quality_language_duration_or_label_filter':True, 'TRAIN_admitted':False}
    write_json(out/'report.json',report)
    print(json.dumps({'PID':guard.process.pid,'report_path':str(out/'report.json'),'groups':groups,
                      'top_rows_SHA256':report['top_rows_SHA256'],'private_bytes':report['private_bytes']}),flush=True)
    source.close(); labels.close()


if __name__=='__main__': main()
