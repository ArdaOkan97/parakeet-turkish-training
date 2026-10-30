"""Isolated, resumable TRAIN candidate preparation. Never imports Torch or evaluates a model."""
import argparse, collections, hashlib, io, json, math, os, pathlib, shutil, sqlite3, struct, sys, time, wave

ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=pathlib.Path(r'C:\Users\ardao\Miniconda3\envs\vlm')
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path[:0]=[str(ROOT/'work/training-env/Lib/site-packages'),str(BASE/'Lib/site-packages'),str(ROOT/'outputs/parakeet_turkish')]
_dll=os.add_dll_directory(str(BASE/'Library/bin'))
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from metrics import normalize_text
from lazy_training_rows_v1 import iter_array_field

OUT=ROOT/'work/data/WorldSpeech_tr_quality_candidate_v1'
META=ROOT/'work/research/data_expansion_metadata_v1'
RAW=ROOT/'work/data/WorldSpeech_tr_TRAIN_candidate_v1'
REPORT=ROOT/'outputs/parakeet_turkish/WorldSpeech_TR_TRAIN_audio_candidate_preparation_v1.json'
REV='f389569345a04eb8ac8fa9405b0d5048ad74b8b3'
TOK='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
FILTER='geometry_ok=1 AND roundtrip_ok=1 AND cer IS NOT NULL AND cer<=0.10 AND mos IS NOT NULL AND mos>=2.0 AND duration<=20.0 AND language="tr"'
POLICY={'kind':'WorldSpeech_TR_TRAIN_candidate_filter_v1','source':'disco-eth/WorldSpeech','converted_revision':REV,
 'split':'train','configuration':'tr_tr','source_CER_max':0.10,'DNSMOS_min':2.0,'SNR_cutoff':None,
 'duration_seconds':[0.5,20.0],'target_U_max':180,'tokenizer_semantic_sha256':TOK,'normalized_tokenizer_roundtrip_required':True,
 'human_label_policy':'Unchanged human_transcript encoded with exact original8199; no teacher relabeling, cropping, or text repair.',
 'audio_policy':'Decode embedded source audio; arithmetic channel mean; scipy resample_poly to16k; SoundFile PCM16 quantization; no augmentation.',
 'frame_policy':'mel=floor(actualPCMframes/160); encoder=ceil(mel/8); perclip fp32 joint=T*(U+1)*8204*4.',
 'max_perclip_joint_bytes':1645131712,'heldout_exclusion':'All fixed DEV/TEST IDs, audio/WAV/canonicalPCM hashes and normalized reference hashes; exclusion only, no predictions or evaluation.',
 'existing_audio_exclusion':'Original74883 canonicalPCM proof, full original178918 TSC including quarantine, currentTRAIN and six-language replay source/audio hashes.',
 'candidate_dedup':'Exact embedded audio, canonicalPCM, WAV and source ID; identical TRAIN text alone is allowed.',
 'source_terms':'CC-BY-NC4.0 and source-specific terms retained; no commercial/release clearance claimed.',
 'selected_without_DEV_TEST_predictions':True,'candidate_admitted_to_model':False,'quality_improvement_tested':False,
 'GPU_used':False,'Torch_imported':False,'disk_reserve_bytes':20000000000,'single_CPU_worker':True}

def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def rec(path):
    p=pathlib.Path(path).resolve();return {'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode()

def atomic_json(path,value):
    p=pathlib.Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def rows(path):
    with pathlib.Path(path).open(encoding='utf-8-sig') as f:
        for line in f:yield json.loads(line)

def verified(path,sha):
    p=pathlib.Path(path);assert digest(p)==sha,(str(p),'changed source');return p

def add_guard(db,row,scope):
    for k in ['audio_sha256','wav_sha256','canonical_pcm_sha256']:
        if row.get(k):db.execute('INSERT OR IGNORE INTO forbidden_audio VALUES(?,?)',(row[k],scope))
    db.execute('INSERT OR IGNORE INTO forbidden_ids VALUES(?)',(row['source_id'],))
    if scope=='heldout':
        for language in {row['language'],'tr'}:
            text=normalize_text(row['text'],language)
            db.execute('INSERT OR IGNORE INTO heldout_text VALUES(?)',(hashlib.sha256(text.encode()).hexdigest(),))

def seed_guard(db):
    parent=ROOT/'outputs/parakeet_turkish/tr_public_v3_preparation.json'
    p=json.loads(verified(parent,'5c220d7fa4fa554b741afa7c565afe3f1500b3de8bfa99de0b1040a450cf4980').read_text())
    labels=verified(ROOT/'outputs/parakeet_turkish/tr_public_v3_tokenizer_acceptance.json','42aa606057ba78232ef930f7317675ef318dddb034bc302e4de2247eedd04e3a')
    count=0
    for item in iter_array_field(labels,'accepted_source_labels_and_audio'):
        db.execute('INSERT OR IGNORE INTO forbidden_audio VALUES(?,?)',(item['canonical_pcm_sha256'],'existingTRAIN'));count+=1
    assert count==74883
    import_proof=ROOT/'outputs/parakeet_turkish/tsc_TRAIN_candidate_full_v1_preparation.json'
    t=json.loads(verified(import_proof,'777ce4e173523cca7ec3bd821b53fe37348344005b11380e27adaf20987bb281').read_text())
    tsc=verified(t['accepted_manifest_path'],t['accepted_manifest_sha256'])
    count_tsc=0
    for row in rows(tsc):add_guard(db,row,'existingTRAIN');count_tsc+=1
    assert count_tsc==178918
    config=json.loads((ROOT/'outputs/parakeet_turkish/runs/adaptive_20261014_TR_uniform_token_population_01/run_config.json').read_text(encoding='utf-8-sig'))
    manifests=[config['tr_manifest'],config['replay_manifest'],config['portuguese_replay_manifest']]
    for path in manifests:
        for row in rows(path):add_guard(db,row,'existingTRAIN_or_replay')
    held=dict(p['heldout_manifests_sha256'])
    pt_path=ROOT/'outputs/parakeet_turkish/Portuguese_FLEURS_retention_preparation_v1.json'
    pt=json.loads(verified(pt_path,'f669cdb63e3b874322c1feb7fb9f6dba63e4a9525d0d1af9a59d47a0017496ed').read_text())
    for key in ['validation_pt','test_pt']:held[pt['manifests'][key]['path']]=pt['manifests'][key]['sha256']
    held_count=0
    for path,sha in held.items():
        for row in rows(verified(path,sha)):
            add_guard(db,row,'heldout');held_count+=1
            # Actual small heldout PCM content is read only for exclusion; no model or predictions.
            with wave.open(row['audio_filepath'],'rb') as f:
                assert (f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getcomptype())==(16000,1,2,'NONE')
                pcm=f.readframes(f.getnframes())
            db.execute('INSERT OR IGNORE INTO forbidden_audio VALUES(?,?)',(hashlib.sha256(pcm).hexdigest(),'heldout'))
    assert 'torch' not in sys.modules
    return {'original_PCM_records':count,'original_TSC_records_including_quarantine':count_tsc,'heldout_manifest_records':held_count,
            'protected_metadata_sources':[rec(parent),rec(labels),rec(import_proof),rec(pt_path)],
            'heldout_manifests':held,'heldout_predictions_or_evaluation_used':False}

def preparation_status(db,status,**extra):
    n,h,pcm=db.execute('SELECT COUNT(*),COALESCE(SUM(frames)/16000./3600.,0),COALESCE(SUM(frames)*2,0) FROM accepted').fetchone()
    value={'kind':'ROOT_WorldSpeech_TRAIN_actual_audio_candidate_preparation_v1','pid':os.getpid(),'status':status,
      'updated_unix':time.time(),'completed_files':db.execute('SELECT COUNT(*) FROM files').fetchone()[0],
      'prepared_candidate_clips':n,'prepared_candidate_hours':h,'PCM_payload_bytes':pcm,
      'candidate_admitted_to_model':False,'GPU_used':False,'Torch_imported':False,**extra}
    atomic_json(OUT/'preparation_status.json',value);return value

def main(args):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'audio').mkdir(exist_ok=True)
    policy=OUT/'candidate_filter_policy.json'
    if policy.exists():assert json.loads(policy.read_text())==POLICY
    else:atomic_json(policy,POLICY)
    summary=json.loads((META/'candidate_metadata_token_index_summary.json').read_text())
    assert summary['all_TRAIN_complete'] and summary['rows']==227669 and summary['tokenizer_semantic_sha256']==TOK
    raw_summary=json.loads((RAW/'staging_status.json').read_text())
    assert raw_summary['status']=='complete' and raw_summary['completed_files']==365 and raw_summary['converted_revision']==REV
    source_db=sqlite3.connect((META/'worldspeech_TRAIN_candidate_metadata.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
    source_db.row_factory=sqlite3.Row;source_db.execute('PRAGMA cache_size=-2048')
    metadata_stat=(META/'worldspeech_TRAIN_candidate_metadata.sqlite').stat()
    db=sqlite3.connect(OUT/'preparation.sqlite');db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS forbidden_audio(sha TEXT PRIMARY KEY,scope TEXT);
      CREATE TABLE IF NOT EXISTS forbidden_ids(source_id TEXT PRIMARY KEY);CREATE TABLE IF NOT EXISTS heldout_text(sha TEXT PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS files(filename TEXT PRIMARY KEY,source_sha TEXT,selected INTEGER,accepted INTEGER,rejected TEXT);
      CREATE TABLE IF NOT EXISTS accepted(source_id TEXT PRIMARY KEY,filename TEXT,row_index INTEGER,embedded_sha TEXT UNIQUE,
        pcm_sha TEXT UNIQUE,wav_sha TEXT UNIQUE,frames INTEGER,manifest TEXT,metadata TEXT,qc TEXT);
      CREATE TABLE IF NOT EXISTS rejected(source_id TEXT PRIMARY KEY,filename TEXT,row_index INTEGER,reason TEXT);''')
    if not db.execute('SELECT 1 FROM config WHERE key="guards_complete"').fetchone():
        with db:
            guard=seed_guard(db);db.execute('INSERT INTO config VALUES(?,?)',('guards_complete',json.dumps(guard,sort_keys=True)))
    candidate_count,candidate_hours=source_db.execute('SELECT COUNT(*),SUM(duration)/3600. FROM rows WHERE '+FILTER).fetchone()
    assert candidate_count==98972
    print(json.dumps({'pid':os.getpid(),'candidate_clips':candidate_count,'candidate_hours':candidate_hours,'CPU_only':True}),flush=True)
    new_files=0
    for file in source_db.execute('SELECT filename,source_sha256,rows FROM files ORDER BY filename'):
        name=file['filename'];assert '/' not in name and '\\' not in name
        done=db.execute('SELECT source_sha FROM files WHERE filename=?',(name,)).fetchone()
        if done:assert done[0]==file['source_sha256'];continue
        if (OUT/'stop.txt').exists():preparation_status(db,'stopped_by_owned_stop_file');return
        if args.max_files and new_files>=args.max_files:preparation_status(db,'pilot_complete',candidate_clips=candidate_count,candidate_hours=candidate_hours);return
        assert shutil.disk_usage(OUT).free>POLICY['disk_reserve_bytes']
        selected={x['row_index']:dict(x) for x in source_db.execute('SELECT * FROM rows WHERE filename=? AND '+FILTER,(name,))}
        path=RAW/name;source_proof=json.loads((RAW/'records'/(name+'.json')).read_text())
        assert source_proof['sha256']==file['source_sha256'] and source_proof['bytes']==path.stat().st_size and source_proof['converted_revision']==REV
        before=(path.stat().st_size,path.stat().st_mtime_ns);rejections=collections.Counter();accepted_count=0;ordinal=0
        with db:
            for batch in pq.ParquetFile(path,pre_buffer=False).iter_batches(batch_size=16,columns=['audio','segment_id'],use_threads=False):
                for source in batch.to_pylist():
                    i=ordinal;ordinal+=1
                    if i not in selected:continue
                    row=selected[i];source_id=row['source_id'];assert source_id==f'disco-eth/WorldSpeech:{REV}:tr_tr:train:{source["segment_id"]}'
                    reason=None
                    if db.execute('SELECT 1 FROM forbidden_ids WHERE source_id=?',(source_id,)).fetchone():reason='protected_source_ID'
                    if db.execute('SELECT 1 FROM heldout_text WHERE sha=?',(row['normalized_sha256'],)).fetchone():reason='fixed_heldout_reference_overlap'
                    try:
                        payload=source['audio']['bytes'];assert isinstance(payload,bytes) and payload
                        embedded=hashlib.sha256(payload).hexdigest()
                        if db.execute('SELECT 1 FROM forbidden_audio WHERE sha=?',(embedded,)).fetchone():reason='protected_embedded_audio_overlap'
                        signal,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
                        assert signal.size and 0<signal.shape[1]<=8 and 8000<=rate<=96000 and np.isfinite(signal).all()
                        source_frames=len(signal);channels=signal.shape[1]
                        assert abs(source_frames/rate-row['duration'])<=.002
                        signal=signal.mean(axis=1)
                        if rate!=16000:
                            factor=math.gcd(int(rate),16000);signal=resample_poly(signal,16000//factor,int(rate)//factor)
                        assert signal.size and np.isfinite(signal).all() and np.any(signal!=0)
                        frames=len(signal);duration=frames/16000.;mel=frames//160;T=(mel+7)//8;U=row['U']
                        assert .5<=duration<=20.001 and 1<=U<=180 and T*(U+1)*8204*4<=POLICY['max_perclip_joint_bytes']
                        labels=list(struct.unpack('<'+'H'*U,row['label_ids']));label_sha=hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()
                        assert 0<=min(labels) and max(labels)<8199 and 8192 not in labels
                        buffer=io.BytesIO();sf.write(buffer,signal,16000,format='WAV',subtype='PCM_16');wav=buffer.getvalue()
                        with wave.open(io.BytesIO(wav),'rb') as f:
                            assert (f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getnframes(),f.getcomptype())==(16000,1,2,frames,'NONE')
                            pcm=f.readframes(frames)
                        pcm_sha=hashlib.sha256(pcm).hexdigest();wav_sha=hashlib.sha256(wav).hexdigest()
                        for value in [pcm_sha,wav_sha]:
                            if db.execute('SELECT 1 FROM forbidden_audio WHERE sha=?',(value,)).fetchone():reason='protected_canonical_audio_overlap'
                        if db.execute('SELECT 1 FROM accepted WHERE embedded_sha=? OR pcm_sha=? OR wav_sha=?',(embedded,pcm_sha,wav_sha)).fetchone():reason='candidate_exact_audio_duplicate'
                        values=np.frombuffer(pcm,dtype='<i2');assert np.any(values!=0)
                        qc={'source_rate':int(rate),'source_channels':channels,'source_frames':source_frames,'source_duration':source_frames/rate,
                            'resampled_float_peak':float(np.max(np.abs(signal))),'RMS':float(np.sqrt(np.mean(signal.astype('float64')**2))),
                            'zero_fraction':float(np.count_nonzero(values==0)/frames),'clipping_fraction':float(np.count_nonzero((values==-32768)|(values==32767))/frames),
                            'source_CER':row['cer'],'SNR':row['snr'],'DNSMOS':row['mos'],'perclip_joint_bytes':T*(U+1)*8204*4}
                        if reason is None:
                            folder=OUT/'audio'/name.removesuffix('.parquet');folder.mkdir(exist_ok=True)
                            audio=folder/f'{i:06d}.wav';tmp=audio.with_suffix('.tmp')
                            with tmp.open('wb') as f:f.write(wav);f.flush();os.fsync(f.fileno())
                            os.replace(tmp,audio)
                            manifest={'dataset':'disco-eth/WorldSpeech','dataset_revision':REV,'dataset_config':'tr_tr','dataset_split':'train',
                              'training_role':'train','dataset_file':name,'source_id':source_id,'sentence_id':None,'original_audio_path':source['audio'].get('path'),
                              'language':'tr','text':row['human_text'],'reference':row['human_text'],'reference_normalized':normalize_text(row['human_text'],'tr'),
                              'audio_filepath':str(audio.resolve()),'audio_sha256':embedded,'wav_sha256':wav_sha,'canonical_pcm_sha256':pcm_sha,
                              'duration':duration,'sample_rate':16000,'source_duration':row['duration'],'source_url':row['source_url'],
                              'source_start_s':row['source_start'],'source_end_s':row['source_end'],'session_date':row['session_date'],
                              'speaker_id':None,'gender':None,'target_tokens':U,'label_ids_sha256':label_sha,'source_transcript_sha256':hashlib.sha256(row['human_text'].encode()).hexdigest()}
                            stat=audio.stat();metadata={'source_id':source_id,'actual_PCM_frames':frames,'valid_mel_frames':mel,'encoder_frames':T,
                              'target_tokens':U,'label_ids_sha256':label_sha,'teacher_feature_values_required_for_TR_preservation':False,
                              'canonical_WAV_stat_at_preparation':{'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}}
                            db.execute('INSERT INTO accepted VALUES('+','.join('?' for _ in range(10))+')',
                              (source_id,name,i,embedded,pcm_sha,wav_sha,frames,canonical(manifest).decode(),canonical(metadata).decode(),canonical(qc).decode()));accepted_count+=1
                    except (RuntimeError,ValueError,AssertionError) as error:
                        reason='audio_decode_or_geometry_failure:'+type(error).__name__+':'+str(error)[:160]
                    if reason is not None:
                        db.execute('INSERT INTO rejected VALUES(?,?,?,?)',(source_id,name,i,reason));rejections[reason]+=1
            assert ordinal==file['rows'] and before==(path.stat().st_size,path.stat().st_mtime_ns)
            assert len(selected)==accepted_count+sum(rejections.values())
            db.execute('INSERT INTO files VALUES(?,?,?,?,?)',(name,file['source_sha256'],len(selected),accepted_count,json.dumps(rejections,sort_keys=True)))
        new_files+=1;status=preparation_status(db,'running',candidate_clips=candidate_count,candidate_hours=candidate_hours,current_file=name)
        if new_files%10==0 or args.max_files:print(json.dumps(status),flush=True)
        assert 'torch' not in sys.modules
    assert db.execute('SELECT COUNT(*) FROM files').fetchone()[0]==365
    exports=[]
    for column,name in [('manifest','candidate_train.jsonl'),('metadata','candidate_frame_metadata.jsonl'),('qc','candidate_audio_QC.jsonl')]:
        target=OUT/name;tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as f:
            for (value,) in db.execute('SELECT '+column+' FROM accepted ORDER BY filename,row_index'):f.write(value.encode()+b'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,target);exports.append(rec(target))
    current=(META/'worldspeech_TRAIN_candidate_metadata.sqlite').stat()
    assert (current.st_size,current.st_mtime_ns)==(metadata_stat.st_size,metadata_stat.st_mtime_ns)
    rejection_counts={k:v for k,v in db.execute('SELECT reason,COUNT(*) FROM rejected GROUP BY reason')}
    final=preparation_status(db,'complete',candidate_clips=candidate_count,candidate_hours=candidate_hours,rejections=rejection_counts)
    final.update(policy=rec(policy),helper=rec(pathlib.Path(__file__)),staged_TRAIN_manifest=rec(RAW/'complete_TRAIN_manifest.json'),
                 metadata_index_summary=rec(META/'candidate_metadata_token_index_summary.json'),exports=exports,
                 exclusion_guard=json.loads(db.execute('SELECT value FROM config WHERE key="guards_complete"').fetchone()[0]),
                 speaker_video_and_perceptual_near_duplicate_disjointness_proven=False,
                 independent_TRAIN_transcript_sanity_pending=True,native_memory_admission_pending=True,
                 source_byte_note='Actual staged SHA records and stable source size/stat bound; audio content actually decoded for every selected candidate.')
    assert final['prepared_candidate_clips']+sum(rejection_counts.values())==candidate_count
    atomic_json(REPORT,final);print(json.dumps(final),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-files',type=int,default=0);args=parser.parse_args()
    assert args.max_files>=0
    try:main(args)
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'preparation_failure.json',{'pid':os.getpid(),'error_type':type(error).__name__,'error':str(error),'updated_unix':time.time(),'GPU_used':False})
        raise
