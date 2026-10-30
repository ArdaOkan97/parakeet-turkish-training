"""Bounded CPU pilot QC; original compressed audio stays on E:, no admission."""
import collections
import datetime
import hashlib
import json
import sqlite3
import unicodedata

from yodas_compressed_audio_v1 import ROOT, check_policy, decode_segment, write_json


def text_key(text):
    return ' '.join(unicodedata.normalize('NFC', text).translate(str.maketrans('Iİ', 'ıi')).lower().split())


def stratum(row):
    duration = '2-6s' if row['duration_s'] < 6 else ('6-12s' if row['duration_s'] < 12 else '12-20s')
    quality = 'DNSMOS2.8-3.2' if row['dnsmos'] < 3.2 else 'DNSMOS3.2+'
    return (duration, quality, row['noise'] or 'unknown', row['reverb'] or 'unknown')


def main():
    check_policy()
    output = ROOT / 'analysis/alania_yodas_preparation_v1'
    status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    if not status.get('pilot_staging_actual_CLOSED0') and status['phase'] != 'compressed_pilot_staged_not_TRAIN_admitted':
        raise ValueError('Wait for actual compressed pilot closure')
    shards = status['pilot_shards']
    if len(shards) != 3:
        raise ValueError('This bounded pilot is declared for three staged shards')
    db = sqlite3.connect('file:' + (ROOT / 'indexes/alania_metadata_candidates_v1.sqlite').as_posix() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA cache_size=-8192')
    summaries, selected = [], []
    for shard in shards:
        archive_receipt = json.loads((ROOT / f'datasets/yodas3/audio/{shard}.tar.receipt.json').read_text(encoding='utf-8'))
        if archive_receipt['revision'] != '81d802f45a1c1e1f5331260484dacf1fbea81981':
            raise ValueError('Wrong pinned compressed source')
        buckets = collections.defaultdict(list)
        text_counts = collections.Counter()
        rows = db.execute('SELECT * FROM candidates WHERE eligible=1 AND shard=? ORDER BY recording_id,start_s,end_s,segment_id', (shard,))
        total_seconds = union_seconds = 0.0
        prior_recording, union_end = None, 0.0
        count = overlapping = 0
        source_recordings, speakers = set(), set()
        for raw in rows:
            row = dict(raw)
            count += 1
            total_seconds += row['duration_s']
            source_recordings.add(row['recording_id'])
            speakers.add(row['speaker'])
            text_counts[text_key(row['transcript'])] += 1
            if row['recording_id'] != prior_recording:
                union_end = row['start_s']
                prior_recording = row['recording_id']
            elif row['start_s'] < union_end - .001:
                overlapping += 1
            union_seconds += max(0.0, row['end_s'] - max(row['start_s'], union_end))
            union_end = max(union_end, row['end_s'])
            buckets[stratum(row)].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda r: (r['pilot_rank'], r['segment_id']))
        strata = sorted(buckets)
        seen_recordings, shard_selected = set(), []
        # Equal-turn strata sampling is diagnostic, not a training distribution.
        while len(shard_selected) < 24:
            progress = False
            for key in strata:
                bucket = buckets[key]
                while bucket and bucket[0]['recording_id'] in seen_recordings:
                    bucket.pop(0)
                if not bucket:
                    continue
                row = bucket.pop(0)
                seen_recordings.add(row['recording_id'])
                row['QC_stratum'] = key
                row['text_repeat_count_within_shard'] = text_counts[text_key(row['transcript'])]
                row['semantic_transcript_verified'] = False
                row['independent_language_verified'] = False
                row['cross_source_audio_dedup_complete'] = False
                row['heldout_exclusion_complete'] = False
                shard_selected.append(row)
                progress = True
                if len(shard_selected) == 24:
                    break
            if not progress:
                raise ValueError('Insufficient distinct recordings for the declared QC sample')
        summaries.append({'shard': shard, 'metadata_candidate_clips': count,
            'summed_segment_hours': total_seconds / 3600, 'within_recording_time_union_hours': union_seconds / 3600,
            'segments_overlapping_prior_span': overlapping, 'recordings': len(source_recordings),
            'per_recording_speaker_clusters': len(speakers), 'available_metadata_strata': len(strata),
            'exact_normalized_repeated_transcript_groups': sum(n > 1 for n in text_counts.values()),
            'extra_rows_with_repeated_transcript': sum(n - 1 for n in text_counts.values() if n > 1),
            'repeated_text_is_not_audio_duplicate_proof': True, 'compressed_archive': archive_receipt})
        selected.extend(shard_selected)
    db.close()
    manifest = output / 'stratified_QC_sample_v1.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    write_json(output / 'stratified_QC_metadata_v1.json', {'per_shard': summaries,
        'sample_manifest_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest()})
    ledger = output / 'stratified_QC_decode_ledger_v1.jsonl'
    observations = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    done = {row['segment_id'] for row in observations}
    if len(done) != len(observations) or not done.issubset({r['segment_id'] for r in selected}):
        raise ValueError('QC resume ledger does not match declared sample')
    excluded = []
    for row in selected:
        if row['segment_id'] in done:
            continue
        try:
            pcm, info = decode_segment(row['shard'], row['recording_id'], row['start_s'], row['duration_s'])
        except ValueError as error:
            if str(error) != 'Compressed recording exceeds bounded temporary-member limit':
                raise
            excluded.append({'segment_id': row['segment_id'], 'recording_id': row['recording_id'],
                'shard': row['shard'], 'decode_verified': False, 'reason': str(error),
                'policy': 'Original 256MiB member bound preserved; no resource retry or admission'})
            print(json.dumps({'QC_excluded': row['segment_id'], 'reason': str(error)}), flush=True)
            continue
        info['segment_id'] = row['segment_id']
        info['fraction_samples_abs_below_0_001'] = sum(abs(x) < .001 for x in pcm) / len(pcm)
        info['fraction_samples_abs_at_least_0_999'] = sum(abs(x) >= .999 for x in pcm) / len(pcm)
        del pcm
        info['decode_verified'] = True
        observations.append(info)
        with ledger.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(info, ensure_ascii=False) + '\n')
        print(json.dumps({'QC_decoded': len(observations), 'total': len(selected), 'segment_id': row['segment_id']}), flush=True)
    report = {'kind': 'Alania_YODAS_stratified_CPU_QC_pilot_v1',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed_decode_mechanics_for_reported_decodes': True, 'sample_segments': len(selected),
        'decoded_segments': len(observations), 'excluded_untested_segments': excluded,
        'different_source_recordings': len({x['recording_id'] for x in selected}),
        'per_shard': summaries, 'sample_manifest': str(manifest),
        'sample_manifest_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'sampling': '24 distinct recordings per shard; equal turns across duration/DNSMOS/noise/reverb strata; deterministic metadata rank',
        'scope': 'Decode and metadata diagnostics only; source labels are machine-produced and not independently certified',
        'accepted_TRAIN_hours': 0, 'semantic_listening_audit_pending': True,
        'independent_language_and_alignment_QC_pending': True,
        'source_audio_dedup_and_heldout_exclusion_pending': True,
        'no_human_quality_labels_inferred_from_PCM_statistics': True,
        'GPU_used': False, 'TEST_predictions_read': False,
        'permanent_WAV_or_PCM_written': False, 'observations': observations,
        'partial_result_recovery': 'First66 successful decodes observed on closed session70142 stdout; their per-clip PCM statistics were not durably saved and are not reconstructed'}
    path = output / 'stratified_QC_report_v1.json'
    write_json(path, report)
    print(json.dumps({'report': str(path), 'SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'segments': len(selected), 'accepted_TRAIN_hours': 0}), flush=True)


if __name__ == '__main__':
    main()
