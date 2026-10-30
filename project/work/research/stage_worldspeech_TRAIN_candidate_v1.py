"""Stage pinned public TRAIN Parquet files for local assessment; no model admission."""
import hashlib, json, os, pathlib, re, shutil, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
META = ROOT / 'work/research/data_expansion_metadata_v1'
DEST = ROOT / 'work/data/WorldSpeech_tr_TRAIN_candidate_v1'
REV = json.loads((META / 'worldspeech_converted_revision.json').read_text())['sha']
RATE = 6 * 1024 * 1024
FILES = [x for x in json.loads((META / 'worldspeech_parquet.json').read_text())['parquet_files'] if x['split']=='train']
assert len(FILES) == 365 and all(x['config']=='tr_tr' for x in FILES)
DEST.mkdir(parents=True, exist_ok=True)
RECORDS = DEST / 'records'; RECORDS.mkdir(exist_ok=True)
started = time.monotonic(); transferred = 0; last_publish = 0; completed = []

def atomic(path, obj):
    assert path.resolve().is_relative_to(DEST.resolve())
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)

def publish(status, current=None, error=None):
    global last_publish
    atomic(DEST/'staging_status.json', {'kind':'ROOT_WorldSpeech_TR_TRAIN_candidate_staging_v1',
        'pid':os.getpid(), 'status':status, 'converted_revision':REV, 'completed_files':len(completed),
        'total_TRAIN_files':len(FILES), 'completed_bytes':sum(x['bytes'] for x in completed),
        'transferred_this_session_bytes':transferred, 'current_file':current, 'error':error,
        'bandwidth_cap_bytes_per_second':RATE, 'updated_unix':time.time(),
        'source_terms':'WorldSpeech CC-BY-NC4.0 and source-specific terms retained; no commercial/release clearance claimed.',
        'candidate_admitted_to_model':False, 'GPU_used':False, 'Torch_imported':False,
        'TEST_files_downloaded':False, 'manifest_sha256':hashlib.sha256((META/'worldspeech_parquet.json').read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
    last_publish = time.monotonic()

def file_record(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        if f.read(4) != b'PAR1': raise ValueError('Invalid Parquet header')
        f.seek(-4,2)
        if f.read(4) != b'PAR1': raise ValueError('Invalid Parquet footer')
        f.seek(0)
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return {'path':str(path.resolve()), 'bytes':path.stat().st_size, 'sha256':h.hexdigest(),
        'converted_revision':REV, 'split':'train'}

try:
    remaining = sum(x['size'] for x in FILES)
    if shutil.disk_usage(DEST).free < remaining + 20_000_000_000: raise RuntimeError('Insufficient local staging capacity')
    publish('running')
    print(json.dumps({'pid':os.getpid(),'destination':str(DEST),'files':len(FILES),'bytes':remaining,'cap_MiB_per_s':6}),flush=True)
    for item in FILES:
        name = item['filename']
        if not re.fullmatch(r'\d+\.parquet',name): raise ValueError('Unexpected TRAIN filename')
        target = DEST/name
        assert target.resolve().is_relative_to(DEST.resolve())
        url = item['url'].replace('refs%2Fconvert%2Fparquet', REV)
        if not url.startswith(f'https://huggingface.co/datasets/disco-eth/WorldSpeech/resolve/{REV}/tr_tr/train/'):
            raise ValueError('Only pinned Turkish TRAIN endpoint allowed')
        if target.exists():
            if target.stat().st_size != item['size']: raise ValueError('Existing staged file size mismatch')
        else:
            part = target.with_suffix('.parquet.partial'); offset = part.stat().st_size if part.exists() else 0
            if offset > item['size']: raise ValueError('Partial file exceeds pinned size')
            if offset < item['size']:
                headers = {'User-Agent':'Local-ASR-TRAIN-candidate-staging/1.0'}
                if offset: headers['Range']=f'bytes={offset}-'
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as r,part.open('ab') as f:
                    if offset and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                        raise ValueError('Resume range was not honored')
                    if not offset and r.status not in (200,206): raise ValueError('Unexpected download status')
                    while True:
                        if (DEST/'stop.txt').exists(): raise RuntimeError('Owned candidate staging stop requested')
                        chunk=r.read(min(1048576,item['size']-f.tell()+1))
                        if not chunk:break
                        if f.tell()+len(chunk)>item['size']: raise ValueError('Pinned size exceeded')
                        f.write(chunk);transferred+=len(chunk)
                        delay=transferred/RATE-(time.monotonic()-started)
                        if delay>0:time.sleep(min(delay,1))
                        if time.monotonic()-last_publish>10:publish('running',name)
                    f.flush();os.fsync(f.fileno())
            if part.stat().st_size!=item['size']: raise ValueError('Incomplete staged file')
            file_record(part)
            os.replace(part,target)
        rec=file_record(target);rec['source_url']=url
        record_path=RECORDS/(name+'.json')
        if record_path.exists() and json.loads(record_path.read_text())!=rec:
            raise ValueError('Existing immutable source record changed')
        atomic(record_path,rec);completed.append(rec)
        if len(completed)%10==0:
            publish('running',name);print(json.dumps({'completed_files':len(completed),'bytes':sum(x['bytes'] for x in completed)}),flush=True)
    atomic(DEST/'complete_TRAIN_manifest.json',{'converted_revision':REV,'files':completed,'audio_decoded':False,'admitted':False})
    publish('complete');print(json.dumps({'complete':True,'files':len(completed)}),flush=True)
except BaseException as exc:
    publish('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
    raise
