"""Stream all source labels into an E-only catalog; no quality/LID thresholds."""
import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import shutil
import sqlite3
import sys
import threading
import time

import psutil
import pyarrow.parquet as pq

from cpu_preparation_guard_v2 import memory, scoped, write_json

ROOT = pathlib.Path(r'E:\gpu_training_scratchpad')
AREA = ROOT / 'analysis/BPE_weak_then_clean_v1'
PROJECT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from metrics import normalize_text


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with pathlib.Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Catalog:
    def __init__(self):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
        for key in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR'):
            scoped(os.environ[key])
        self.process = psutil.Process()
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.checked = 0
        self.state = {'owner': '/root', 'task': 'ASR-18', 'pid': os.getpid(),
                      'started_at_utc': now(), 'quality_filters': [], 'language_filters': [],
                      'TRAIN_admitted': False, 'GPU_used': False,
                      'audio_decoded': False, 'TEST_predictions_read': False}
        self.db_path = ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite'
        self.db = sqlite3.connect(self.db_path)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA temp_store=MEMORY')
        self.db.execute('PRAGMA cache_size=-8192')
        self.db.execute('CREATE TABLE IF NOT EXISTS clips (ordinal INTEGER PRIMARY KEY, '
                        'unique_id TEXT UNIQUE, dataset TEXT, original_split TEXT, source_id TEXT, '
                        'shard TEXT, row_index INTEGER, start REAL, end REAL, duration REAL, '
                        'text TEXT, text_raw TEXT, text_spoken TEXT, original_flags_json TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS files (file TEXT PRIMARY KEY, source_sha256 TEXT, '
                        'rows_seen INTEGER, rows_retained INTEGER, retained_seconds REAL, exclusions_json TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS exclusions (unique_id TEXT PRIMARY KEY, '
                        'dataset TEXT, reason TEXT, duration REAL)')
        self.db.commit()
        self.heldout_sources = set()
        self.heldout_text = set()
        self.bindings = []
        self.state['runtime_PCM_heldout_exclusion_still_required'] = True
        self.publish(status='starting')
        threading.Thread(target=self.watchdog, daemon=True).start()

    def publish(self, **fields):
        self.state.update(fields)
        self.state['updated_at_utc'] = now()
        write_json(AREA / 'broad_metadata_status_v1.json', self.state)

    def watchdog(self):
        while True:
            time.sleep(2)
            try:
                m = memory()
                if (m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000
                        or self.process.memory_info().private >= 2_000_000_000):
                    write_json(AREA / 'broad_metadata_resource_stop_v1.json',
                               {'at_utc': now(), 'pid': os.getpid(), 'memory': m})
                    os._exit(75)
            except Exception:
                os._exit(76)

    def guard(self):
        if time.monotonic() - self.checked < 2:
            return
        self.checked = time.monotonic()
        if shutil.disk_usage(ROOT).free < 100_000_000_000:
            raise RuntimeError('E100GBreserve')
        m = memory()
        while m['commit_ratio'] >= .90:
            self.publish(status='yielding_for_memory', memory=m)
            time.sleep(2)
            m = memory()
        self.state['memory'] = m

    def fixed_heldouts(self):
        roots = [PROJECT / 'work/data', PROJECT / 'work/data/portuguese_retention_v1']
        ids = set()
        for base in roots:
            for path in sorted(base.glob('*.jsonl')):
                if not path.name.startswith(('validation', 'test')):
                    continue
                count = 0
                with path.open(encoding='utf-8-sig') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if 'prediction' in row or 'hypothesis' in row:
                            raise ValueError('Only source heldout manifests permitted')
                        if row.get('dataset_split', row.get('split')) not in ('validation', 'test'):
                            raise ValueError('Unexpected heldout source split')
                        sid = row['source_id']
                        ids.add(sid)
                        self.heldout_sources.add(sid)
                        self.heldout_text.add(normalize_text(row['reference'], 'tr'))
                        count += 1
                self.bindings.append({'path': str(path), 'sha256': sha(path), 'rows': count})
        if len(ids) != 1050:
            raise ValueError('Expected complete1050 fixed heldout source identities')
        self.state['fixed_heldout_unique_source_ids'] = len(ids)

    def kiraat_heldouts(self, jobs):
        sources = set()
        text_keys = set()
        for job in jobs:
            p = pathlib.Path(job['destination'])
            while not p.exists() or not p.with_suffix(p.suffix + '.receipt.json').exists():
                self.guard()
                self.publish(status='waiting_for_pinned_KIRAAT_holdout_metadata', waiting_for=p.name)
                time.sleep(10)
            for batch in pq.ParquetFile(p).iter_batches(batch_size=256,
                    columns=['source_id', 'text', 'text_raw', 'text_spoken'], use_threads=False):
                for row in batch.to_pylist():
                    sources.add(row['source_id'])
                    for field in ('text', 'text_raw', 'text_spoken'):
                        if row[field]:
                            text_keys.add(normalize_text(row[field], 'tr'))
            self.bindings.append({'path': str(p), 'sha256': job['sha256'], 'role': 'KIRAAT_published_heldout'})
        self.heldout_sources.update(sources)
        self.heldout_text.update(text_keys)
        self.state.update(kiraat_heldout_source_ids=len(sources), kiraat_heldout_text_keys=len(text_keys))
        write_json(AREA / 'heldout_metadata_bindings_v1.json',
                   {'files': self.bindings, 'source_ids': sorted(self.heldout_sources),
                    'normalized_text_keys': sorted(self.heldout_text),
                    'TEST_predictions_read': False, 'complete_audio_exclusion_claimed': False})

    def add(self, uid, dataset, split, source_id, shard, row_index, start, end, duration,
            text, text_raw, text_spoken, flags):
        reason = None
        if (not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end, duration))
                or start < 0 or end <= start or duration <= 0):
            reason = 'unusable_nonpositive_or_nonfinite_time_span'
        elif not isinstance(text, str) or not text.strip():
            reason = 'unusable_empty_supplied_label'
        elif source_id in self.heldout_sources:
            reason = 'heldout_source_identity'
        elif any(normalize_text(t, 'tr') in self.heldout_text for t in (text, text_raw, text_spoken) if t):
            reason = 'exact_normalized_heldout_reference_overlap'
        if reason:
            self.db.execute('INSERT INTO exclusions VALUES (?,?,?,?)', (uid, dataset, reason,
                            duration if isinstance(duration, (int, float)) and math.isfinite(duration) else 0))
            return reason
        self.db.execute('INSERT INTO clips (unique_id,dataset,original_split,source_id,shard,row_index,'
                        'start,end,duration,text,text_raw,text_spoken,original_flags_json) '
                        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (uid, dataset, split, source_id, shard, row_index, start, end, duration,
                         text, text_raw, text_spoken, json.dumps(flags, ensure_ascii=False)))
        return None

    def already(self, name, digest):
        row = self.db.execute('SELECT source_sha256 FROM files WHERE file=?', (name,)).fetchone()
        if row and row[0] != digest:
            raise ValueError('Previously committed metadata source identity changed')
        return bool(row)

    def alania(self):
        p = ROOT / 'indexes/alania_metadata_candidates_v1.sqlite'
        digest = sha(p)
        if self.already('alania_all_metadata', digest):
            return
        source = sqlite3.connect(p.as_uri() + '?mode=ro&immutable=1', uri=True)
        source.row_factory = sqlite3.Row
        source.execute('PRAGMA cache_size=-8192')
        exclusions = collections.Counter()
        seen = kept = 0
        seconds = 0.
        for raw in source.execute('SELECT segment_id,recording_id,shard,start_s,end_s,duration_s,'
                                  'transcript,eligible,reasons FROM candidates ORDER BY rowid'):
            row = dict(raw)
            self.guard()
            reason = self.add('alania:' + row['segment_id'], 'alania_yodas3', 'all_metadata',
                              row['recording_id'], row['shard'], seen, row['start_s'], row['end_s'],
                              row['duration_s'], row['transcript'], row['transcript'], None,
                              {'historical_eligible_IGNORED': row['eligible'],
                               'historical_reasons_IGNORED': row['reasons']})
            seen += 1
            if reason:
                exclusions[reason] += 1
            else:
                kept += 1
                seconds += row['duration_s']
            if seen % 10000 == 0:
                self.publish(status='indexing_all_Alania_labels', source_rows_seen=seen,
                             current_source_rows_retained=kept)
        if seen != 835267:
            raise ValueError('Pinned all-Alania row count differs')
        self.db.execute('INSERT INTO files VALUES (?,?,?,?,?,?)',
                        ('alania_all_metadata', digest, seen, kept, seconds, json.dumps(exclusions)))
        self.db.commit()
        source.close()
        self.summary()

    def kiraat_file(self, job):
        if self.already(job['path'], job['sha256']):
            return
        p = pathlib.Path(job['destination'])
        while not p.exists() or not p.with_suffix(p.suffix + '.receipt.json').exists():
            self.guard()
            self.publish(status='waiting_for_next_pinned_KIRAAT_file', waiting_for=p.name)
            time.sleep(15)
        columns = ['id', 'source_id', 'start', 'end', 'duration', 'text', 'text_raw', 'text_spoken',
                   'recommended', 'exclusion_reasons', 'flags', 'source_flags', 'duplicate_of', 'split']
        seen = kept = 0
        seconds = 0.
        exclusions = collections.Counter()
        split = p.name.split('-')[0]
        for batch in pq.ParquetFile(p).iter_batches(batch_size=256, columns=columns, use_threads=False):
            self.guard()
            for row in batch.to_pylist():
                # Use the supplied orthographic transcript; keep raw/spoken variants separately.
                reason = self.add('kiraat:' + row['id'], 'kiraat', split, row['source_id'],
                                  str(p), seen, row['start'], row['end'], row['duration'], row['text'],
                                  row['text_raw'], row['text_spoken'],
                                  {k: row[k] for k in ('recommended', 'exclusion_reasons', 'flags',
                                                       'source_flags', 'duplicate_of', 'split')})
                seen += 1
                if reason:
                    exclusions[reason] += 1
                else:
                    kept += 1
                    seconds += row['duration']
        self.db.execute('INSERT INTO files VALUES (?,?,?,?,?,?)',
                        (job['path'], job['sha256'], seen, kept, seconds, json.dumps(exclusions)))
        self.db.commit()
        self.summary()

    def summary(self):
        totals = self.db.execute('SELECT SUM(rows_seen),SUM(rows_retained),SUM(retained_seconds),COUNT(*) '
                                 'FROM files').fetchone()
        self.publish(status='metadata_file_committed', source_rows_seen=totals[0],
                     metadata_retained_rows=totals[1], metadata_retained_hours=(totals[2] or 0) / 3600,
                     committed_metadata_files=totals[3])
        exclusions = collections.Counter()
        groups = {}
        for name, seen, kept, seconds, raw_exclusions in self.db.execute(
                'SELECT file,rows_seen,rows_retained,retained_seconds,exclusions_json FROM files'):
            exclusions.update(json.loads(raw_exclusions))
            group = ('alania_yodas3', 'all_metadata') if name == 'alania_all_metadata' else (
                'kiraat', pathlib.PurePosixPath(name).name.split('-')[0])
            value = groups.setdefault(group, {'dataset': group[0], 'split': group[1], 'clips': 0, 'hours': 0.})
            value['clips'] += kept
            value['hours'] += seconds / 3600
        write_json(AREA / 'broad_metadata_report_v1.json',
                   {**self.state, 'catalog_path': str(self.db_path),
                    'exclusion_counts': dict(exclusions), 'groups': list(groups.values()),
                    'complete_audio_exclusion_claimed': False, 'TRAIN_admitted': False})

    def run(self):
        plan = json.loads((AREA / 'preparation_plan_v1.json').read_text())
        jobs = [j for j in plan['download_jobs'] if j['repository'] == 'serdarcaglar/kiraat']
        self.fixed_heldouts()
        self.kiraat_heldouts([j for j in jobs if pathlib.Path(j['path']).name.startswith(('dev-', 'test-'))])
        self.alania()
        for job in sorted((j for j in jobs if pathlib.Path(j['path']).name.startswith(('train-', 'rest-'))),
                          key=lambda j: j['path']):
            self.kiraat_file(job)
        self.db.execute('CREATE INDEX IF NOT EXISTS clips_source ON clips(source_id)')
        self.db.execute('CREATE INDEX IF NOT EXISTS clips_dataset ON clips(dataset,original_split)')
        self.db.commit()
        self.summary()
        self.publish(status='completed_metadata_only', metadata_complete=True, actual_exit_code=0,
                     finished_at_utc=now())
        self.db.close()
        write_json(AREA / 'broad_metadata_actual_CLOSED0_v1.json',
                   {**self.state, 'catalog_path': str(self.db_path), 'catalog_sha256': sha(self.db_path)})


if __name__ == '__main__':
    worker = None
    try:
        worker = Catalog()
        worker.run()
    except BaseException as error:
        if worker is not None:
            worker.db.rollback()
            worker.publish(status='stopped', failure=type(error).__name__ + ': ' + str(error))
        raise
