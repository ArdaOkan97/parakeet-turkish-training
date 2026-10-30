"""Resume committed scalar geometry from a bound checkpoint, with atomic retry I/O."""
import argparse
import hashlib
import os
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
        end = time.monotonic() + count / 262144
        while time.monotonic() < end:
            self.wait()
            time.sleep(min(.025, max(0, end-time.monotonic())))


def checked(pin):
    path = Path(pin['path'])
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.stat().st_size < 8_000_000
    raw = path.read_bytes()
    assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256']
    value = read_json(path)
    assert value is not None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-plan', required=True)
    args = parser.parse_args()
    assert all(os.environ[key] == '1' for key in
               ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    guard = MetadataGuard('incoming_label_geometry', heavy=False, maximum_bytes=128_000_000)
    guard.wait()
    plan = read_json(args.resume_plan)
    assert plan['owner'] == '/root/cpu_data_preparation'
    old_exit = checked(plan['prior_actual_exit'])
    assert old_exit['actual_exit_code'] == 1 and old_exit['actual_tool_chunk_id']
    checkpoint = checked(plan['resume_checkpoint'])
    baseline = checked(plan['original_pinned_envelope'])
    assert checkpoint['baseline_frontier'] == baseline['pinned_label_frontier']
    groups = dict(checkpoint['groups_all_committed'])
    frontier = checkpoint['observed_committed_frontier']
    assert groups['rows'] == frontier >= baseline['pinned_label_frontier']
    first = list(checkpoint['first_U_blocker_ordinals'])
    labels_path = ROOT / 'indexes/BPE_broad_labels_8711_v1.sqlite'
    labels = sqlite3.connect(labels_path.as_uri()+'?mode=ro', uri=True, timeout=2)
    labels.execute('PRAGMA cache_size=-1024')
    labels.execute('PRAGMA temp_store=MEMORY')
    labels.execute('PRAGMA query_only=ON')
    assert labels.execute('SELECT frontier FROM progress WHERE id=1').fetchone()[0] >= frontier
    start = time.monotonic()
    status = AREA / 'incoming_label_geometry_status_v1.json'

    def publish(phase):
        write_json(status, {'kind': 'ASR18_incoming_committed_label_geometry_v1', 'at_utc': now(),
            'PID': guard.process.pid, 'status': phase, 'observer_revision': 2,
            'baseline_frontier': baseline['pinned_label_frontier'],
            'observed_committed_frontier': frontier, 'groups_all_committed': groups,
            'first_U_blocker_ordinals': first,
            'readiness_blockers_not_filters': groups['count_U_gt_458'] > 0 or groups['count_U_ge_1024'] > 0,
            'whole_Alania_baseline_preserved': True, 'KIRAAT_and_full_corpus_still_partial': True,
            'resume_checkpoint': plan['resume_checkpoint'], 'rows_excluded': 0,
            'TRAIN_admitted': False, 'actual_paced_metadata_bytes': guard.bytes})

    publish('resumed_exact_published_checkpoint_no_rescan')
    while time.monotonic()-start < 86400:
        guard.wait()
        target = labels.execute('SELECT frontier FROM progress WHERE id=1').fetchone()[0]
        if target > frontier:
            end = min(target, frontier+8192)
            guard.pace((end-frontier)*32)
            rows = labels.execute('SELECT ordinal,target_tokens,conservative_estimated_joint_bytes '
                'FROM labels WHERE ordinal>? AND ordinal<=? ORDER BY ordinal', (frontier, end)).fetchall()
            assert len(rows) == end-frontier
            for ordinal, length, dense in rows:
                assert dense % (8716*4) == 0
                groups['rows'] += 1
                groups['sum_U'] += length
                groups['MAX_U'] = max(groups['MAX_U'], length)
                groups['count_U_gt_458'] += length > 458
                groups['count_U_ge_1024'] += length >= 1024
                groups['count_U_plus_1_gt_1024'] += length+1 > 1024
                groups['MAX_estimated_T_times_U_plus_1'] = max(
                    groups['MAX_estimated_T_times_U_plus_1'], dense//(8716*4))
                if length > 458 and len(first) < 8:
                    first.append({'ordinal': ordinal, 'U': length})
            frontier = end
            publish('new_commits_checked')
        else:
            publish('caught_up_to_latest_committed_frontier')
            actual = read_json(AREA / 'BPE_labels_owned_tool_actual_CLOSED0_v3.json')
            if actual is not None and actual['actual_exit_code'] == 0:
                publish('completed_committed_label_observation_actual_writer_closed')
                break
            time.sleep(2)
    else:
        publish('bounded_24hour_observation_finished')
    labels.close()
    assert not any(name in sys.modules for name in ('torch', 'av', 'transformers', 'soundfile'))


if __name__ == '__main__':
    main()
