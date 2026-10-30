"""Read pinned TRAIN numeric Parquet columns with bounded HTTP ranges; no audio decode."""
import argparse, concurrent.futures, hashlib, io, json, math, os, pathlib, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = pathlib.Path(r'C:\Users\ardao\Miniconda3\envs\vlm')
sys.path.append(str(BASE / 'Lib/site-packages'))
_dll = os.add_dll_directory(str(BASE / 'Library/bin'))
import pyarrow.parquet as pq

META = ROOT / 'work/research/data_expansion_metadata_v1'
REV = json.loads((META / 'worldspeech_converted_revision.json').read_text())['sha']
COLUMNS = ['duration', 'cer', 'snr', 'dnsmos_ovr']

class Ranged(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.fetched = url, size, 0, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        p = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        if not 0 <= p <= self.size: raise ValueError('Out-of-file seek')
        self.pos = p
        return p
    def read(self, n=-1):
        if n < 0: raise ValueError('Unbounded network read forbidden')
        n = min(n, self.size - self.pos)
        if not n: return b''
        if n > 2_000_000 or self.fetched + n > 4_000_000: raise ValueError('Per-file range budget exceeded')
        start = self.pos
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{start+n-1}', 'User-Agent': 'Local-ASR-projected-metadata/1.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                raise ValueError('Server did not honor exact ranged request')
            data = r.read(n + 1)
            if len(data) != n: raise ValueError('Range size mismatch')
        self.pos += n
        self.fetched += n
        return data

def projected(item):
    name = item['filename']
    target = META / 'projected_numeric' / (name + '.json')
    if target.exists(): return json.loads(target.read_text())
    url = item['url'].replace('refs%2Fconvert%2Fparquet', REV)
    f = Ranged(url, item['size'])
    parquet = pq.ParquetFile(f, pre_buffer=False, buffer_size=0)
    table = parquet.read(columns=COLUMNS, use_threads=False)
    if table.column_names != COLUMNS: raise ValueError('Required numeric columns absent')
    rows = table.to_pylist()
    duration = [float(x['duration']) for x in rows]
    if not all(math.isfinite(x) and x > 0 for x in duration): raise ValueError('Bad duration')
    def stat(key):
        a = sorted(float(x[key]) for x in rows if x[key] is not None and math.isfinite(float(x[key])))
        return {'n':len(a), 'sum':math.fsum(a), 'min':a[0] if a else None, 'max':a[-1] if a else None,
                'median':a[len(a)//2] if a else None}
    result = {'converted_revision':REV, 'split':'train', 'filename':name, 'url':url, 'rows':len(rows),
        'parquet_declared_rows':parquet.metadata.num_rows, 'duration_seconds':math.fsum(duration),
        'numeric_columns':{key:stat(key) for key in COLUMNS}, 'rows_duration_half_to_30s':sum(.5 <= x <= 30 for x in duration),
        'seconds_duration_half_to_30s':math.fsum(x for x in duration if .5 <= x <= 30),
        'range_bytes_fetched':f.fetched, 'audio_column_materialized':False, 'Torch_imported':'torch' in sys.modules,
        'note':'Parquet footer/range reads may contain incidental adjacent bytes. No complete audio object is requested or decoded.'}
    if result['rows'] != result['parquet_declared_rows']: raise ValueError('Projected row mismatch')
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    os.replace(temporary, target)
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument('--limit', type=int, default=3); p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    files = [x for x in json.loads((META / 'worldspeech_parquet.json').read_text())['parquet_files'] if x['split']=='train']
    selected = files[:args.limit] if args.limit > 0 else files
    completed, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(projected, x):x for x in selected}
        for future in concurrent.futures.as_completed(futures):
            try: completed.append(future.result())
            except Exception as e: errors.append({'filename':futures[future]['filename'], 'error':repr(e)})
            n = len(completed)+len(errors)
            if n % 25 == 0: print(json.dumps({'completed':len(completed), 'failed':len(errors), 'total':len(selected)}), flush=True)
    result = {'kind':'ROOT_WorldSpeech_TRAIN_numeric_projection_v1', 'converted_revision':REV, 'requested_files':len(selected),
        'expected_all_TRAIN_files':len(files), 'completed_files':len(completed), 'errors':errors,
        'rows':sum(x['rows'] for x in completed), 'duration_seconds':math.fsum(x['duration_seconds'] for x in completed),
        'duration_hours':math.fsum(x['duration_seconds'] for x in completed)/3600,
        'duration_half_to_30s_hours':math.fsum(x['seconds_duration_half_to_30s'] for x in completed)/3600,
        'range_bytes_fetched':sum(x['range_bytes_fetched'] for x in completed),
        'projected_columns':COLUMNS, 'all_TRAIN_complete':len(completed)==len(files) and not errors,
        'no_audio_decoding':True, 'Torch_imported':'torch' in sys.modules, 'DEV_TEST_read':False,
        'source_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    (META / 'projected_TRAIN_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return int(bool(errors))

if __name__ == '__main__': raise SystemExit(main())
