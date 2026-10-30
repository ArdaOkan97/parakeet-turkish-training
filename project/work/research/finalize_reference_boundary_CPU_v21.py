"""Record an observed tool exit and seal the committed reference frontier.

This reads cached index metadata only. It does not decode audio, load ASR, or
start another interval. The actual exit arguments come from the owned tool.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from reference_frontier_codec_v21 import SCHEMA, header_bytes, FrontierEncoder, FrontierDecoder

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import psutil
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced
import run_random_tiny_language_CPU_v7 as gating
from run_large_v3_sanity_cpu_v6 import hard_process_memory_job


class MetadataGuard(gating.Guard):
    def __init__(self):
        base.Guard.__init__(self, 'cached_reference_frontier_receipt_v21')
        self.proof_verified = False
        self.last_poll = 0.
        self.last_snapshot = self.last_reason = None
        self.paused_seconds = 0.
        self.file_bytes_read = 0
        self.io_rate = 1048576

    def pace(self, count):
        if self.file_bytes_read+count > 30_000_000:
            raise RuntimeError('Declared 30MB metadata receipt budget reached')
        return super().pace(count)


def canonical_bytes(row):
    return (json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8')


def main():
    session, pid, chunk, code = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    assert (session, pid, code) == (39255, 24292, 0)
    area = ROOT/'analysis/CPU_validation_v21'
    base.AREA = paced.AREA = area
    guard = MetadataGuard()
    guard.wait()
    job = hard_process_memory_job()
    if psutil.pid_exists(pid):
        assert not any('extend_audio_references_next10k_CPU_v21.py' in value for value in psutil.Process(pid).cmdline())
    status = json.loads((area/'next10k_real_audio_references_CPU_v21_status.json').read_text())
    assert status['pid'] == pid and status['phase'] == 'completed_declared_next10k_reference_interval'
    report = json.loads((area/'bounded_next10k_reference_completed_v21.json').read_text())
    assert report['actual_WAV_PCM_verified_reference_clips'] == 130000
    plan_path = area/'bounded_next10k_reference_plan_v21.json'
    plan = json.loads(plan_path.read_text())
    assert hashlib.sha256(Path(plan['source']).read_bytes()).hexdigest() == plan['source_SHA']
    path = ROOT/'indexes/accepted_audio_reference_v4.sqlite'
    db = sqlite3.connect('file:'+path.as_posix()+'?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-2048')
    total = db.execute('SELECT COUNT(*) FROM references_audio').fetchone()[0]
    assert total == 130000
    assert db.execute('SELECT COUNT(*) FROM candidate_batch_checks WHERE batch_id=12').fetchone()[0] == 186
    assert db.execute('SELECT 1 FROM comparisons_complete WHERE batch_id=12').fetchone()
    digest = hashlib.sha256()
    canonical_digest = hashlib.sha256()
    encoder = FrontierEncoder()
    decoder = FrontierDecoder()
    frontier_file = area/'committed_reference_frontier_130000_v21.exact.bin'
    temporary = frontier_file.with_suffix('.bin.writing')
    probes=[];encoded_bytes=0;roundtrip_rows=0
    with temporary.open('wb') as out:
        header=header_bytes();buffer=bytearray(header);digest.update(header)
        probes.append((0,header));encoded_bytes=len(header)
        for row in db.execute('SELECT ordinal,source_id,dataset,duration_s,status,PCM_i16_sha256,frames,spectral_usable FROM references_audio ORDER BY ordinal'):
            data=encoder.encode(row)
            restored=decoder.consume(data)
            assert len(restored)==1 and canonical_bytes(restored[0])==canonical_bytes(row),'Lossless frontier roundtrip failed'
            canonical_digest.update(canonical_bytes(row))
            digest.update(data)
            if roundtrip_rows%10000==0:probes.append((encoded_bytes,data))
            encoded_bytes+=len(data);roundtrip_rows+=1;buffer.extend(data)
            if len(buffer)>=65536:
                guard.pace(len(buffer));out.write(buffer);buffer.clear()
        if buffer:guard.pace(len(buffer));out.write(buffer)
        out.flush();os.fsync(out.fileno())
    assert roundtrip_rows==130000
    temporary.replace(frontier_file)
    assert frontier_file.stat().st_size==encoded_bytes
    with frontier_file.open('rb') as sample:
        for offset,expected in probes:
            guard.pace(len(expected));sample.seek(offset);assert sample.read(len(expected))==expected
    prior_closure=ROOT/'analysis/CPU_validation_v20/actual_next10k_reference_CLOSED0_v20.json'
    prior=json.loads(prior_closure.read_text(encoding='utf-8'))
    assert prior['exit_code']==0 and prior['actual_reference_clips']==120000
    assert Path(prior['committed_reference_frontier_path']).exists()
    schema_path=area/'compact_exact_frontier_schema_v21.json'
    base.write_json(schema_path,SCHEMA)
    exact = spectral = 0
    checks_digest = hashlib.sha256()
    for sid, details in db.execute('SELECT segment_id,details FROM candidate_batch_checks WHERE batch_id=12 ORDER BY segment_id'):
        data = canonical_bytes((sid, details))
        guard.pace(len(data))
        checks_digest.update(data)
        record = json.loads(details)
        exact += bool(record['exact_accepted_PCM_matches'])
        spectral += bool(record['conservative_spectral_triage_matches'])
    states = dict(db.execute('SELECT status,COUNT(*) FROM candidate_features GROUP BY status'))
    complete = [row[0] for row in db.execute('SELECT batch_id FROM comparisons_complete ORDER BY batch_id')]
    checks_per_batch = dict(db.execute('SELECT batch_id,COUNT(*) FROM candidate_batch_checks GROUP BY batch_id ORDER BY batch_id'))
    assert checks_per_batch == {i:186 for i in range(13)}
    cumulative_checks = sum(checks_per_batch.values())
    db.close()
    interval = area/'reference_interval_ordinals_v21.u32'
    interval_data = guard.read_file(interval)
    value = {
        'kind': 'actual_next10k_reference_CLOSED0_v21',
        'recorded_at_utc': base.now(), 'exit_code': code,
        'session_id': session, 'pid': pid, 'tool_chunk': chunk,
        'admitted_source_path': plan['source'], 'admitted_source_SHA256': plan['source_SHA'],
        'admitted_phase_plan_path': str(plan_path),
        'actual_reference_clips': total,
        'reference_training_duration_h': report['actual_audio_reference_hours'],
        'spectral_usable_reference_clips': report['spectral_reference_clips_usable_under_current_min8frames'],
        'reference_integrity_or_availability_errors': report['unavailable_or_integrity_error_reference_rows'],
        'interval_in_immutable_global_order': [120000, 130000],
        'interval_ordinals_path': str(interval),
        'interval_ordinals_SHA256': hashlib.sha256(interval_data).hexdigest(),
        'index_path': str(path), 'index_size_bytes_at_closure': path.stat().st_size,
        'committed_reference_frontier_path': str(frontier_file),
        'committed_reference_frontier_SHA256': digest.hexdigest(),
        'frontier_representation': ['ordinal','source_id','dataset','duration_s','status','PCM_i16_sha256','frames','spectral_usable'],
        'vectors_remain_in_resumable_E_index': True,
        'frontier_file_format': 'accepted_reference_exact_frontier_binary_v1',
        'frontier_schema_path': str(schema_path),
        'frontier_decoder_source': str(PROJECT/'work/research/reference_frontier_codec_v21.py'),
        'frontier_decoder_source_SHA256': hashlib.sha256((PROJECT/'work/research/reference_frontier_codec_v21.py').read_bytes()).hexdigest(),
        'frontier_canonical_JSONL_equivalent_SHA256': canonical_digest.hexdigest(),
        'lossless_encoder_independent_decoder_roundtrip_verified_rows': roundtrip_rows,
        'bounded_on_disk_record_probe_count': len(probes),
        'compact_frontier_encoded_bytes': encoded_bytes,
        'metadata_receipt_maximum_bytes': 30_000_000,
        'metadata_budget_counts_compact_encoded_bytes_plus_checks_and_probes': True,
        'previous_120k_actual_closure_preserved': str(prior_closure),
        'previous_120k_frontier_preserved': prior['committed_reference_frontier_path'],
        'previous_120k_frontier_preserved_SHA256': prior['committed_reference_frontier_SHA256'],
        'source_candidate_feature_status_counts': states,
        'actual_candidate_reference_comparisons_this_interval': 186,
        'actual_candidate_batch_checks_cumulative': cumulative_checks,
        'actual_candidate_checks_per_completed_batch': checks_per_batch,
        'existing_accepted_pool_reference_clips_total': 352768,
        'existing_accepted_pool_training_duration_h': 675.0964019097223,
        'reference_training_duration_coverage_fraction': report['actual_audio_reference_hours']/675.0964019097223,
        'reference_hours_are_existing_accepted_audio_coverage_not_new_TRAIN': True,
        'completed_reference_comparison_batch_ids': complete,
        'candidate_checks_this_interval_SHA256': checks_digest.hexdigest(),
        'exact_PCM_match_candidates_this_interval': exact,
        'spectral_match_candidates_this_interval': spectral,
        'aggregate_paced_read_bytes_this_interval': report['aggregate_paced_read_bytes'],
        'metadata_receipt_paced_read_bytes': guard.file_bytes_read,
        'clip_reference_coverage_fraction': total/352768,
        'full675h_audio_exclusion_complete': False,
        'no_match_does_not_prove_uniqueness': True,
        '17text_overlap_and3longmatch_quarantines_unchanged': True,
        'tiny_v8_raw_flags_unchanged': True,
        'accepted_TRAIN_hours': 0, 'GPU_used': False,
        'permanent_audio_written': False,
    }
    closure_path = area/'actual_next10k_reference_CLOSED0_v21.json'
    base.write_json(closure_path, value)
    guard.status(phase='observed_actual_CLOSED0_and_frontier_sealed', actual_reference_clips=total,
                 exact_PCM_match_candidates_this_interval=exact, spectral_match_candidates_this_interval=spectral)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
