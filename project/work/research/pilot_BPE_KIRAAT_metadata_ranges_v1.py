"""Four pinned train/rest Parquet metadata projections; original audio untouched.

Requires the GPU owner's expiring metadata_ranges role permit. It deliberately
cannot satisfy the complete-archive acquisition receipt or TRAIN admission.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

from ASR18_data_io_guard_v2 import DataIOGuard, ROOT, AREA
from cpu_preparation_guard_v2 import now, write_json

ROLE = 'metadata_ranges'
CACHE = ROOT / 'scratch/BPE_KIRAAT_metadata_ranges_v1'
REPORT = AREA / 'metadata_range_pilot_v1'
COLUMNS = ['id', 'source_id', 'start', 'end', 'duration', 'text', 'text_raw',
           'text_spoken', 'recommended', 'exclusion_reasons', 'flags',
           'source_flags', 'duplicate_of', 'split']
FILES = ['data/rest-00001-of-00137.parquet', 'data/rest-00002-of-00137.parquet',
         'data/train-00001-of-00592.parquet', 'data/train-00002-of-00592.parquet']


class RangeFile(io.RawIOBase):
    def __init__(self, job, guard):
        import requests
        from huggingface_hub import get_token
        self.job = job; self.guard = guard; self.offset = 0
        self.directory = CACHE / job['sha256']; self.directory.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        token = get_token()
        assert token, 'Authenticated KIRAAT access required'
        self.headers = {'Authorization': 'Bearer ' + token, 'Accept-Encoding': 'identity'}
        self.url = f"https://huggingface.co/datasets/{job['repository']}/resolve/{job['revision']}/{job['path']}"
        self.ranges = []; self.next_at = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.offset

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.offset + offset if whence == 1 else self.job['size'] + offset
        assert 0 <= target <= self.job['size'], 'Range outside pinned original size'
        self.offset = target; return target

    def pace(self, count):
        # The frozen heavy-role guard supplies permission and resource bounds;
        # this stricter local pacer keeps the metadata transport <=256 KiB/s.
        self.guard.pace(count)
        deadline = max(self.next_at, time.monotonic()) + count / 262144
        while time.monotonic() < deadline:
            self.guard.wait(); time.sleep(min(.025, max(0, deadline - time.monotonic())))
        self.next_at = deadline

    def read(self, n=-1):
        if n < 0 or n > 16 * 1024 * 1024:
            raise ValueError('Unbounded or over16MiB read forbidden; audio columns prohibited')
        n = min(n, self.job['size'] - self.offset)
        if not n: return b''
        start = self.offset; end = start + n - 1
        target = self.directory / f'{start}_{n}.range'
        receipt_path = target.with_suffix('.range.receipt.json')
        blocks = []; digest = hashlib.sha256()
        if target.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
            assert receipt['source_revision'] == self.job['revision'] and receipt['source_full_SHA256'] == self.job['sha256']
            with target.open('rb', buffering=0) as stream:
                while True:
                    self.pace(65536); block = stream.read(65536)
                    if not block: break
                    digest.update(block); blocks.append(block)
            assert sum(map(len, blocks)) == n and digest.hexdigest() == receipt['range_SHA256']
        else:
            partial = target.with_suffix('.range.part')
            # Preserve interrupted cache artifacts. A fresh attempt has a unique
            # suffix; never overwrite/delete a prior partial or original file.
            attempt = 0
            while partial.exists():
                attempt += 1; partial = target.with_suffix(f'.range.part{attempt}')
            headers = {**self.headers, 'Range': f'bytes={start}-{end}'}
            self.guard.wait()
            with self.session.get(self.url, headers=headers, stream=True, timeout=(20, 30)) as response:
                assert response.status_code == 206, 'Pinned metadata range was not honored'
                assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{self.job["size"]}', 'Pinned range extent differs'
                assert response.headers.get('Content-Encoding', 'identity') == 'identity'
                with partial.open('xb', buffering=0) as stream:
                    iterator = response.iter_content(chunk_size=65536)
                    consumed = 0
                    while consumed < n:
                        self.pace(min(65536, n - consumed))
                        block = next(iterator)
                        if not block: continue
                        assert consumed + len(block) <= n
                        stream.write(block); digest.update(block); blocks.append(block); consumed += len(block)
                    stream.flush(); os.fsync(stream.fileno())
            assert consumed == n
            partial.replace(target)
            write_json(receipt_path, {'source_repository': self.job['repository'], 'source_revision': self.job['revision'],
                'source_path': self.job['path'], 'source_size': self.job['size'], 'source_full_SHA256': self.job['sha256'],
                'range_start': start, 'range_bytes': n, 'range_SHA256': digest.hexdigest(),
                'HTTP_206_Content_Range_verified': True, 'whole_original_SHA256_verified': False, 'at_utc': now()})
        self.offset += n
        self.ranges.append({'start': start, 'bytes': n, 'SHA256': digest.hexdigest(), 'cache': str(target)})
        return b''.join(blocks)

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data; return len(data)

    def close(self):
        if not self.closed: self.session.close()
        super().close()


def main():
    guard = DataIOGuard(ROLE, heavy=True, maximum_bytes=32_000_000)
    import pyarrow.parquet as pq
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    plan = json.loads((AREA / 'preparation_plan_v1.json').read_text(encoding='utf-8'))
    jobs = [next(j for j in plan['download_jobs'] if j['repository'] == 'serdarcaglar/kiraat' and j['path'] == name) for name in FILES]
    assert len(jobs) == 4 and all(j['revision'] == 'f8afe95abedc19ec6392c77cf470f5d3dcd2788f' for j in jobs)
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORT / 'pinned_manifest.json'
    manifest = {'pinned_before_network': True, 'files': jobs, 'columns': COLUMNS,
                'maximum_transport_bytes': 32_000_000, 'maximum_transport_bytes_s': 262144,
                'whole_archive_acquisition_separate': True, 'catalog_not_modified': True,
                'audio_column_requested': False, 'quality_filters': [], 'TRAIN_admitted': False}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8-sig')); assert old == manifest
    else: write_json(manifest_path, manifest)
    results = []
    for job in jobs:
        result_path = REPORT / (Path(job['path']).stem + '.result.json')
        if result_path.exists():
            results.append(json.loads(result_path.read_text())); continue
        guard.wait(); before = guard.bytes
        target = REPORT / (Path(job['path']).stem + '.metadata.jsonl')
        partial = target.with_suffix('.jsonl.part')
        attempt = 0
        while partial.exists() or target.exists():
            attempt += 1
            target = REPORT / (Path(job['path']).stem + f'.metadata_attempt{attempt}.jsonl')
            partial = target.with_suffix('.jsonl.part')
        row_count = 0
        with RangeFile(job, guard) as stream:
            parquet = pq.ParquetFile(stream, pre_buffer=False, memory_map=False)
            expected_rows = parquet.metadata.num_rows
            with partial.open('x', encoding='utf-8') as output:
                for batch in parquet.iter_batches(batch_size=256, columns=COLUMNS, use_threads=False):
                    guard.wait()
                    for row in batch.to_pylist():
                        assert 'audio' not in row and set(row) == set(COLUMNS)
                        output.write(json.dumps({'row_index': row_count, **row}, ensure_ascii=False) + '\n')
                        row_count += 1
                    output.flush()
                os.fsync(output.fileno())
            assert row_count == expected_rows
            result = {'source_path': job['path'], 'source_revision': job['revision'], 'source_full_SHA256': job['sha256'],
                      'whole_original_SHA256_verified': False, 'rows': row_count, 'expected_rows': expected_rows,
                      'projected_columns': COLUMNS, 'metadata_output': str(target), 'ranges': stream.ranges,
                      'paced_transport_bytes': guard.bytes - before, 'raw_flags_preserved': True, 'quality_filters': [],
                      'audio_column_requested': False, 'catalog_not_modified': True, 'TRAIN_admitted': False}
        partial.replace(target); write_json(result_path, result); results.append(result)
        guard.publish('metadata_range_file_complete', completed_files=len(results), metadata_rows=sum(x['rows'] for x in results))
    result = {'kind': 'ASR18_metadata_only_range_pilot_v1', 'PID': os.getpid(), 'actual_exit_code': 0,
              'completed_files': len(results), 'metadata_rows': sum(x['rows'] for x in results), 'files': results,
              'paced_transport_bytes': guard.bytes, 'own_private_bytes': guard.process.memory_info().private,
              'at_utc': now(), 'no_CPU_model_imports': True, 'audio_column_requested': False, 'catalog_not_modified': True,
              'whole_archive_acquisition_still_required': True, 'TRAIN_admitted': False}
    write_json(REPORT / 'result.json', result); guard.publish('completed_metadata_range_pilot', actual_exit_code=0)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        # Never emit signed redirect URLs or authentication headers.
        REPORT.mkdir(parents=True, exist_ok=True)
        write_json(REPORT / 'failure.json', {'at_utc': now(), 'exception_class': type(error).__name__,
                   'safe_reason': str(error) if isinstance(error, AssertionError) else None,
                   'originals_and_partials_preserved': True, 'TRAIN_admitted': False})
        print(json.dumps({'status': 'stopped', 'exception_class': type(error).__name__}), flush=True)
        sys.exit(1)
