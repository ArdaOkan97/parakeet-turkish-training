"""CPU-only compressed shard index and bounded on-demand segment decoder.

Original tar/WebM bytes are retained. Only a requested compressed recording is
temporarily materialized on E: for accurate-seek FFmpeg decoding; PCM is returned
in memory and never saved as a WAV pool. This is not a training admission step.
"""
import array, datetime, hashlib, json, math, os, pathlib, re, sqlite3
import subprocess, tarfile, tempfile, time

ROOT=pathlib.Path(r'E:\gpu_training_scratchpad').resolve()
FFMPEG=pathlib.Path(r'C:\Users\ardao\Miniconda3\Library\bin\ffmpeg.exe')
FFPROBE=FFMPEG.with_name('ffprobe.exe')
MAX_MEMBER_BYTES=256*1024*1024
MAX_SEGMENT_SECONDS=20

def scoped(path):
    p=pathlib.Path(path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError('Path outside designated E: root')
    return p

def write_json(path,value):
    p=scoped(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def check_policy():
    policy=json.loads((ROOT/'storage_config.json').read_text(encoding='utf-8-sig'))
    if pathlib.Path(policy['storage_root']).resolve()!=ROOT or policy['allow_fallback_outside_root']:
        raise ValueError('Wrong storage policy')
    if not scoped(os.environ['TEMP']).is_relative_to(ROOT): raise ValueError('TEMP outside root')

def index_shard(shard):
    if not re.fullmatch(r'\d{4}',shard): raise ValueError('Invalid shard')
    archive=scoped(ROOT/f'datasets/yodas3/audio/{shard}.tar')
    receipt=json.loads(archive.with_suffix('.tar.receipt.json').read_text(encoding='utf-8'))
    if receipt['bytes']!=archive.stat().st_size or receipt['revision']!='81d802f45a1c1e1f5331260484dacf1fbea81981':
        raise ValueError('Pinned source receipt mismatch')
    index=scoped(ROOT/f'indexes/yodas3_tar_{shard}.sqlite')
    db=sqlite3.connect(index);db.execute('PRAGMA journal_mode=TRUNCATE')
    db.execute('CREATE TABLE IF NOT EXISTS recordings (recording_id TEXT PRIMARY KEY, member_name TEXT, offset INTEGER, bytes INTEGER)')
    count=0
    with tarfile.open(archive,'r:') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.webm'): continue
            name=pathlib.PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not re.fullmatch(r'[a-f0-9]{32}',name.stem):
                raise ValueError('Unsafe tar member name')
            if member.offset_data<0 or member.offset_data+member.size>archive.stat().st_size:
                raise ValueError('Tar member exceeds archive')
            db.execute('INSERT OR REPLACE INTO recordings VALUES (?,?,?,?)',(name.stem,member.name,member.offset_data,member.size))
            count+=1
    db.commit();stored=db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0];db.close()
    if count!=stored: raise ValueError('Duplicate recording IDs in source tar')
    return {'shard':shard,'recordings':count,'archive_bytes':receipt['bytes'],'archive_sha256':receipt['sha256'],'index':str(index)}

def decode_segment(shard,recording_id,start_s,duration_s):
    check_policy()
    if not re.fullmatch(r'\d{4}',shard) or not re.fullmatch(r'[a-f0-9]{32}',recording_id): raise ValueError('Unsafe identity')
    if not math.isfinite(start_s) or not math.isfinite(duration_s) or start_s<0 or not 0<duration_s<=MAX_SEGMENT_SECONDS:
        raise ValueError('Requested segment out of bounds')
    db=sqlite3.connect(f'file:{(ROOT/f"indexes/yodas3_tar_{shard}.sqlite").as_posix()}?mode=ro',uri=True)
    row=db.execute('SELECT offset,bytes FROM recordings WHERE recording_id=?',(recording_id,)).fetchone();db.close()
    if row is None: raise ValueError('Recording absent from pinned tar')
    offset,size=row
    if size>MAX_MEMBER_BYTES: raise ValueError('Compressed recording exceeds bounded temporary-member limit')
    temporary=None;h=hashlib.sha256();before=time.perf_counter()
    try:
        # No tar extraction API: copy exactly the verified member byte extent.
        with tempfile.NamedTemporaryFile(prefix='yodas-member-',suffix='.webm',dir=scoped(ROOT/'temp'),delete=False) as output:
            temporary=pathlib.Path(output.name)
            with scoped(ROOT/f'datasets/yodas3/audio/{shard}.tar').open('rb') as archive:
                archive.seek(offset);remaining=size
                while remaining:
                    block=archive.read(min(4*1024*1024,remaining))
                    if not block: raise ValueError('Truncated source extent')
                    output.write(block);h.update(block);remaining-=len(block)
        command=[str(FFMPEG),'-nostdin','-hide_banner','-loglevel','error','-threads','1',
                 '-ss',format(start_s,'.9f'),'-i',str(temporary),'-t',format(duration_s,'.9f'),
                 '-map','0:a:0','-vn','-sn','-dn','-filter_threads','1','-ac','1','-ar','16000',
                 '-f','f32le','-acodec','pcm_f32le','pipe:1']
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90,check=True,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if len(result.stdout)%4: raise ValueError('Non-float-aligned PCM output')
        samples=array.array('f');samples.frombytes(result.stdout)
        expected=round(duration_s*16000)
        if abs(len(samples)-expected)>80 or not samples or not all(math.isfinite(x) for x in samples):
            raise ValueError('Invalid decoded sample count or nonfinite PCM')
        info={'shard':shard,'recording_id':recording_id,'start_s':start_s,'requested_duration_s':duration_s,
              'decoded_sample_rate':16000,'decoded_channels':1,'decoded_samples':len(samples),
              'decoded_duration_s':len(samples)/16000,'PCM_bytes_in_memory':len(result.stdout),
              'peak_absolute_value':max(abs(x) for x in samples),'RMS':math.sqrt(sum(x*x for x in samples)/len(samples)),
              'compressed_member_bytes':size,'compressed_member_sha256':h.hexdigest(),
              'PCM_sha256':hashlib.sha256(result.stdout).hexdigest(),'elapsed_seconds':time.perf_counter()-before,
              'permanent_PCM_or_WAV_written':False,'transcoded_archive':False,'FFmpeg':str(FFMPEG),
              'seek':'input accurate seek default; output limited to requested duration'}
        return samples,info
    finally:
        # Only this invocation's freshly created temporary compressed member.
        if temporary is not None and temporary.exists(): temporary.unlink()

def validate_pilot(shard,count=12,report_path=None):
    check_policy();index=index_shard(shard)
    db=sqlite3.connect(f'file:{(ROOT/"indexes/alania_metadata_candidates_v1.sqlite").as_posix()}?mode=ro',uri=True)
    columns=['segment_id','recording_id','shard','start_s','end_s','duration_s','transcript','noise','reverb','speaker']
    rows=db.execute('SELECT '+','.join(columns)+' FROM candidates WHERE eligible=1 AND shard=? ORDER BY pilot_rank',(shard,))
    chosen=[];seen=set()
    for row in rows:
        item=dict(zip(columns,row))
        if item['recording_id'] in seen: continue
        seen.add(item['recording_id']);chosen.append(item)
        if len(chosen)==count:break
    db.close();observations=[]
    for row in chosen:
        samples,info=decode_segment(shard,row['recording_id'],row['start_s'],row['duration_s'])
        # Audio is discarded from memory after the check; source stays compressed.
        del samples
        observations.append({**row,**info})
        print(json.dumps({'decoded_pilot_segments':len(observations),'requested_segments':count,
                          'segment_id':row['segment_id'],'seconds':info['decoded_duration_s']},ensure_ascii=False),flush=True)
    # Repeat a decode to verify deterministic bytes on this actual installed path.
    first=chosen[0];samples,repeat=decode_segment(shard,first['recording_id'],first['start_s'],first['duration_s']);del samples
    if repeat['PCM_sha256']!=observations[0]['PCM_sha256']:raise ValueError('Repeated decoder output differed')
    result={'kind':'YODAS_original_compressed_on_demand_CPU_validation_v1',
            'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,
            'archive':index,'decoded_segments':len(observations),'different_recordings':len(seen),
            'repeated_first_decode_same_PCM':True,'bounded_temp_member_bytes':MAX_MEMBER_BYTES,
            'max_segment_seconds':MAX_SEGMENT_SECONDS,'permanent_WAV_copy':False,'GPU_used':False,
            'Torch_imported':'torch' in __import__('sys').modules,'TEST_read':False,'TRAIN_admitted':False,
            'semantic_transcript_quality_verified':False,'audio_dedup_and_heldout_exclusion_complete':False,
            'observations':observations}
    path=report_path or ROOT/'analysis/alania_yodas_preparation_v1/on_demand_decode_validation.json';write_json(path,result)
    print(json.dumps({'passed':True,'decoded_segments':len(observations),'report':str(path)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--shard',required=True);parser.add_argument('--segments',type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.segments<=20:raise ValueError('Bounded CPU pilot only')
    validate_pilot(args.shard,args.segments)
