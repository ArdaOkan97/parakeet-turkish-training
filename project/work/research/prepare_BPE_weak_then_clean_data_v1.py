"""Pinned, resumable acquisition for ASR-18; one bounded CPU download stream."""
import ctypes
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import sys
import threading
import time

import psutil
import requests
from huggingface_hub import get_token

from cpu_preparation_guard_v2 import memory, scoped, write_json

ROOT = pathlib.Path(r'E:\gpu_training_scratchpad')
AREA = ROOT / 'analysis/BPE_weak_then_clean_v1'
PROJECT = pathlib.Path(__file__).resolve().parents[2]
STATUS = AREA / 'acquisition_status_v1.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ResourceStop(RuntimeError):
    pass


class Preparation:
    def __init__(self):
        cfg = json.loads((ROOT / 'storage_config.json').read_text(encoding='utf-8-sig'))
        assert pathlib.Path(cfg['storage_root']).resolve() == ROOT.resolve()
        assert not cfg['allow_fallback_outside_root']
        for name in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR', 'PYTHONPYCACHEPREFIX'):
            scoped(os.environ[name])
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
        assert all(os.environ[k] == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
        self.process = psutil.Process()
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.plan = json.loads((AREA / 'preparation_plan_v1.json').read_text(encoding='utf8'))
        self.checked = 0.0
        self.started = time.monotonic()
        self.received = 0
        self.completed = []
        self.state = {'owner': '/root', 'task': 'ASR-18', 'pid': os.getpid(),
                      'started_at_utc': now(), 'actual_GPU_used': False, 'training_started': False,
                      'download_streams': 1, 'quality_filters': [], 'language_filters': [],
                      'new_download_budget_bytes': self.plan['new_download_bytes'],
                      'status': 'starting', 'completed_files': 0, 'received_this_session_bytes': 0}
        self.publish()
        self.guard(force=True, reserve=self.plan['new_download_bytes'])
        threading.Thread(target=self.watchdog, daemon=True).start()

    def watchdog(self):
        while True:
            time.sleep(2)
            try:
                m = memory()
                private = self.process.memory_info().private
                if (m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000
                        or private >= 2_000_000_000):
                    write_json(AREA / 'acquisition_resource_stop_v1.json',
                               {'at_utc': now(), 'pid': os.getpid(), 'memory': m,
                                'own_private_bytes': private, 'own_process_only': True})
                    os._exit(75)
            except Exception:
                os._exit(76)

    def publish(self, **fields):
        self.state.update(fields)
        self.state.update(updated_at_utc=now(), received_this_session_bytes=self.received,
                          completed_files=len(self.completed))
        write_json(STATUS, self.state)

    def guard(self, force=False, reserve=0):
        if not force and time.monotonic() - self.checked < 2:
            return
        self.checked = time.monotonic()
        m = memory()
        private = self.process.memory_info().private
        free = shutil.disk_usage(ROOT).free
        if free - reserve < 100_000_000_000:
            raise ResourceStop('E_100GB_reserve')
        if private >= 2_000_000_000:
            raise ResourceStop('own_private_2GB_bound')
        if m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000:
            raise ResourceStop('critical_system_memory')
        lease = json.loads((PROJECT / 'work/gpu_training_owner.json').read_text(encoding='utf-8-sig'))
        if lease.get('actual_GPU_running') or lease.get('actual_training_optimizer_running'):
            raise ResourceStop('yield_download_before_ROOT_GPU_work')
        while m['commit_ratio'] >= .90:
            self.publish(status='yielding_for_system_commit', memory=m, own_private_bytes=private)
            time.sleep(2)
            m = memory()
            if m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000:
                raise ResourceStop('critical_system_memory_during_yield')
        self.state.update(memory=m, own_private_bytes=private, free_E_bytes=free)

    def download(self, job):
        destination = scoped(job['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = job['sha256']
        if not expected:
            raise ValueError('Pinned LFS SHA256 required')
        receipt_path = destination.with_suffix(destination.suffix + '.receipt.json')
        if destination.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
            if not (destination.stat().st_size == job['size'] and receipt['sha256'] == expected
                    and receipt['revision'] == job['revision']):
                raise ValueError('Existing complete artifact does not match pinned receipt')
            return {'path': job['path'], 'reused': True, 'sha256': expected}
        partial = destination.with_suffix(destination.suffix + '.part')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > job['size']:
            raise ValueError('Partial exceeds pinned size')
        digest = hashlib.sha256()
        if offset:
            with partial.open('rb') as source:
                for block in iter(lambda: source.read(2 * 1024 * 1024), b''):
                    self.guard()
                    digest.update(block)
        self.publish(status='downloading', current_repository=job['repository'],
                     current_file=job['path'], current_size_bytes=job['size'], current_offset_bytes=offset)
        if offset < job['size']:
            token = get_token() if job['repository'] == 'serdarcaglar/kiraat' else None
            headers = {'Accept-Encoding': 'identity'}
            if token:
                headers['Authorization'] = 'Bearer ' + token
            if offset:
                headers['Range'] = f'bytes={offset}-'
            url = f"https://huggingface.co/datasets/{job['repository']}/resolve/{job['revision']}/{job['path']}"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                if response.status_code not in ([206] if offset else [200, 206]):
                    raise RuntimeError(f'HTTP_{response.status_code}_for_{job["repository"]}/{job["path"]}')
                if offset and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Resume range not honored')
                last = time.monotonic()
                with partial.open('ab' if offset else 'wb') as out:
                    for block in response.iter_content(chunk_size=2 * 1024 * 1024):
                        self.guard()
                        if not block:
                            continue
                        if offset + len(block) > job['size']:
                            raise RuntimeError('Download exceeds pinned size')
                        out.write(block)
                        digest.update(block)
                        offset += len(block)
                        self.received += len(block)
                        if self.received > self.plan['new_download_bytes']:
                            raise ResourceStop('Explicit_download_budget_exceeded')
                        if time.monotonic() - last > 10:
                            out.flush()
                            self.publish(status='downloading', current_offset_bytes=offset,
                                         measured_session_MBps=self.received / max(1, time.monotonic() - self.started) / 1e6)
                            last = time.monotonic()
                    out.flush()
                    os.fsync(out.fileno())
        if offset != job['size'] or digest.hexdigest() != expected:
            raise RuntimeError('Pinned_size_or_SHA256_mismatch_partial_preserved')
        partial.replace(destination)
        receipt = {**job, 'completed_at_utc': now(), 'sha256': digest.hexdigest(),
                   'bytes': offset, 'owner': '/root', 'task': 'ASR-18', 'quality_filter_applied': False}
        write_json(receipt_path, receipt)
        return {'path': job['path'], 'sha256': expected, 'reused': False}

    def run(self):
        jobs = self.plan['download_jobs']
        # Two representative training files and published holdouts first for schema/exclusion preparation.
        front = [j for j in jobs if j['repository'] == 'serdarcaglar/kiraat' and
                 (pathlib.PurePosixPath(j['path']).name.startswith(('train-00000-', 'rest-00000-', 'dev-', 'test-')))]
        ordered = front + [j for j in jobs if j not in front]
        for job in ordered:
            self.guard(force=True)
            item = self.download(job)
            self.completed.append(item)
            write_json(AREA / 'acquisition_completed_files_v1.json', self.completed)
            self.publish(status='file_complete')
            print(json.dumps({'completed_files': len(self.completed), 'file': job['path']}), flush=True)
        self.publish(status='completed', finished_at_utc=now())
        write_json(AREA / 'acquisition_actual_CLOSED0_v1.json', {**self.state, 'actual_exit_code': 0})


if __name__ == '__main__':
    preparation = None
    try:
        preparation = Preparation()
        preparation.run()
    except BaseException as error:
        # Exception strings from HTTP libraries can contain signed URLs; report the class only.
        safe = str(error) if isinstance(error, (ResourceStop, ValueError, RuntimeError)) else type(error).__name__
        if preparation is not None:
            preparation.publish(status='stopped', failure=safe, partials_preserved=True, unchanged_retry=False)
        print(json.dumps({'status': 'stopped', 'reason': safe}), flush=True)
        sys.exit(75 if isinstance(error, ResourceStop) else 1)
