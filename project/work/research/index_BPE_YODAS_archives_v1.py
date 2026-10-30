"""Header-only bounded indexes for pinned downloaded YODAS archives; no decoding."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import psutil
from cpu_preparation_guard_v2 import ROOT, memory, now, scoped, write_json
from stage_yodas_additional_cpu_v2 import index_archive

AREA = ROOT / 'analysis/BPE_weak_then_clean_v1'


class Guard:
    def __init__(self):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
        for name in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR'):
            scoped(os.environ[name])
        self.process = psutil.Process()
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.checked = 0
        self.state = {'owner': '/root', 'task': 'ASR-18', 'pid': os.getpid(),
                      'started_at_utc': now(), 'audio_decoded': False, 'GPU_used': False,
                      'quality_filters': [], 'language_filters': [], 'TRAIN_admitted': False}

    def status(self, **fields):
        self.state.update(fields, updated_at_utc=now())
        write_json(AREA / 'YODAS_archive_index_status_v1.json', self.state)

    def wait(self):
        if time.monotonic() - self.checked < 2:
            return
        self.checked = time.monotonic()
        while True:
            m = memory()
            assert self.process.memory_info().private < 2_000_000_000
            assert m['commit_ratio'] < .95 and m['commit_headroom_bytes'] >= 2_000_000_000
            assert shutil.disk_usage(ROOT).free >= 100_000_000_000
            if m['commit_ratio'] < .90:
                self.state['memory'] = m
                return
            self.status(status='yielding_for_memory', memory=m)
            time.sleep(2)


def main():
    guard = Guard()
    plan = json.loads((AREA / 'preparation_plan_v1.json').read_text())
    jobs = sorted([j for j in plan['download_jobs'] + plan['reused_archives']
                   if j['repository'] == 'espnet/yodas3'], key=lambda j: j['path'])
    done = []
    for job in jobs:
        shard = Path(job['path']).stem
        archive = Path(job['destination'])
        receipt = archive.with_suffix(archive.suffix + '.receipt.json')
        while not archive.exists() or not receipt.exists():
            guard.wait()
            acquisition = json.loads((AREA / 'acquisition_status_v1.json').read_text())
            if acquisition['status'] == 'stopped':
                raise RuntimeError('Acquisition stopped; preserve index frontier without retry')
            guard.status(status='waiting_for_pinned_archive', current_shard=shard,
                         completed_archives=len(done))
            time.sleep(10)
        item = json.loads(receipt.read_text(encoding='utf-8-sig'))
        assert item['sha256'] == job['sha256'] and item['revision'] == job['revision']
        assert archive.stat().st_size == job['size']
        target = ROOT / f'indexes/yodas3_tar_{shard}.sqlite'
        reused = target.exists()
        if not reused:
            guard.status(status='indexing_TAR_headers', current_shard=shard)
            index_archive(shard, guard)
        db = sqlite3.connect(target.as_uri() + '?mode=ro', uri=True)
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        count, low, high = db.execute('SELECT COUNT(*),MIN(offset),MAX(offset+bytes) FROM recordings').fetchone()
        assert count > 0 and low >= 0 and high <= job['size']
        db.close()
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        done.append({'shard': shard, 'archive_sha256': job['sha256'], 'index': str(target),
                     'index_sha256': digest, 'recordings': count, 'reused_index': reused})
        write_json(AREA / 'YODAS_archive_index_progress_v1.json', done)
        guard.status(status='archive_index_complete', current_shard=shard,
                     completed_archives=len(done), recordings_indexed=sum(r['recordings'] for r in done))
    guard.status(status='completed', actual_exit_code=0, completed_archives=len(done))
    write_json(AREA / 'YODAS_archive_indexes_actual_CLOSED0_v1.json', {**guard.state, 'indexes': done})


if __name__ == '__main__':
    main()
