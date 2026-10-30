"""Inactive TRAIN-only label/geometry/quality index; no model or audio decoding."""
import hashlib, json, math, os, pathlib, sqlite3, struct, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=pathlib.Path(r'C:\Users\ardao\Miniconda3\envs\vlm')
sys.path[:0]=[str(ROOT/'work/training-env/Lib/site-packages'),str(BASE/'Lib/site-packages'),str(ROOT/'outputs/parakeet_turkish')]
os.environ['TOKENIZERS_PARALLELISM']='false'
_dll=os.add_dll_directory(str(BASE/'Library/bin'))
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from metrics import normalize_text

OUT=ROOT/'work/research/data_expansion_metadata_v1'
RAW=ROOT/'work/data/WorldSpeech_tr_TRAIN_candidate_v1'
EXPECTED='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
REV=json.loads((OUT/'worldspeech_converted_revision.json').read_text())['sha']
TOKENIZER=ROOT/'outputs/parakeet_turkish/checkpoint/tokenizer.json'
tokenizer=Tokenizer.from_file(str(TOKENIZER))
semantic=hashlib.sha256(json.dumps(json.loads(tokenizer.to_str()),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
assert semantic==EXPECTED and tokenizer.get_vocab_size()==8199 and tokenizer.token_to_id('<blank>')==8192
assert 'torch' not in sys.modules

def record(path):
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

config=json.loads((ROOT/'outputs/parakeet_turkish/runs/adaptive_20261014_TR_uniform_token_population_01/run_config.json').read_text(encoding='utf-8-sig'))
fixture_counts={'cv':0,'fleurs':0,'tsc':0}
with pathlib.Path(config['tr_manifest']).open(encoding='utf-8') as rows,pathlib.Path(config['tr_frame_metadata']).open(encoding='utf-8') as metadata:
    for line,meta in zip(rows,metadata,strict=True):
        row=json.loads(line);item=json.loads(meta)
        group='tsc' if row['dataset']=='issai/Turkish_Speech_Corpus' else ('cv' if 'commonvoice' in row['dataset'].lower() else 'fleurs')
        if fixture_counts[group]>=32:continue
        labels=tokenizer.encode(row['text'],add_special_tokens=False).ids
        assert row['source_id']==item['source_id'] and len(labels)==item['target_tokens']
        assert hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()==item['label_ids_sha256']
        fixture_counts[group]+=1
        if all(x==32 for x in fixture_counts.values()):break
assert all(x==32 for x in fixture_counts.values())
fixture={'kind':'ROOT_exact_original8199_CPU_label_fixture_v1','tokenizer_semantic_sha256':semantic,
    'tokenizer':record(TOKENIZER),'actual_source_label_hash_matches':fixture_counts,
    'TRAIN_label_policy':'Preserve original human text for encoding; shared Turkish normalization is used only for scoring/roundtrip and normalized hashes.',
    'Torch_imported':False,'GPU_used':False,'DEV_TEST_read':False}
(OUT/'exact_original8199_actual_label_fixture_v1.json').write_text(json.dumps(fixture,indent=2)+'\n')

db=sqlite3.connect(OUT/'worldspeech_TRAIN_candidate_metadata.sqlite')
db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
db.executescript('''
CREATE TABLE IF NOT EXISTS files(filename TEXT PRIMARY KEY,source_sha256 TEXT NOT NULL,rows INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS rows(filename TEXT NOT NULL,row_index INTEGER NOT NULL,source_id TEXT NOT NULL,
 human_text TEXT NOT NULL,normalized_sha256 TEXT NOT NULL,label_ids BLOB NOT NULL,U INTEGER NOT NULL,
 duration REAL NOT NULL,T_upper_estimate INTEGER NOT NULL,cer REAL,snr REAL,mos REAL,
 source_url TEXT,source_start REAL,source_end REAL,session_date TEXT,language TEXT,
 geometry_ok INTEGER NOT NULL,roundtrip_ok INTEGER NOT NULL,PRIMARY KEY(filename,row_index));
CREATE INDEX IF NOT EXISTS source_id_index ON rows(source_id);
CREATE INDEX IF NOT EXISTS normalized_index ON rows(normalized_sha256);
''')

COLS=['human_transcript','segment_id','duration','cer','snr','dnsmos_ovr','source_url','source_start_s','source_end_s','session_date','language']
files=[x for x in json.loads((OUT/'worldspeech_parquet.json').read_text())['parquet_files'] if x['split']=='train']
processed=[];missing=[]
def finite(x):return float(x) if x is not None and math.isfinite(float(x)) else None
print(json.dumps({'pid':os.getpid(),'fixture':fixture_counts,'tokenizer_semantic':semantic}),flush=True)
for item in files:
    name=item['filename'];path=RAW/name;proof=RAW/'records'/(name+'.json')
    if not path.exists() or not proof.exists():missing.append(name);continue
    rec=json.loads(proof.read_text());assert rec['converted_revision']==REV and rec['split']=='train'
    assert rec['path']==str(path.resolve()) and rec['bytes']==item['size']==path.stat().st_size
    existing=db.execute('SELECT source_sha256,rows FROM files WHERE filename=?',(name,)).fetchone()
    if existing:
        assert existing[0]==rec['sha256'];processed.append(name);continue
    before=(path.stat().st_size,path.stat().st_mtime_ns)
    table=pq.ParquetFile(path,pre_buffer=False).read(columns=COLS,use_threads=False)
    assert table.column_names==COLS
    batch=[]
    for i,row in enumerate(table.to_pylist()):
        raw=row['human_transcript'] or '';assert isinstance(raw,str)
        normalized=normalize_text(raw,'tr');labels=tokenizer.encode(raw,add_special_tokens=False).ids
        duration=finite(row['duration']);assert duration is not None and duration>0
        valid=bool(labels) and len(labels)<=180 and all(0<=x<8199 and x!=8192 for x in labels)
        roundtrip=normalize_text(tokenizer.decode(labels,skip_special_tokens=True),'tr')==normalized
        # Conservative metadata estimate only; actual resampled PCM/header/frame verification remains mandatory.
        mel_upper=math.ceil(math.ceil(duration*16000)/160);t_upper=(mel_upper+7)//8
        source=f"disco-eth/WorldSpeech:{REV}:tr_tr:train:{row['segment_id']}"
        batch.append((name,i,source,raw,hashlib.sha256(normalized.encode()).hexdigest(),struct.pack('<'+'H'*len(labels),*labels),len(labels),duration,t_upper,
            finite(row['cer']),finite(row['snr']),finite(row['dnsmos_ovr']),row['source_url'],finite(row['source_start_s']),finite(row['source_end_s']),
            row['session_date'],row['language'],int(valid and .5<=duration<=30 and bool(normalized)),int(roundtrip)))
    assert before==(path.stat().st_size,path.stat().st_mtime_ns)
    with db:
        db.executemany('INSERT INTO rows VALUES('+','.join('?' for _ in range(19))+')',batch)
        db.execute('INSERT INTO files VALUES(?,?,?)',(name,rec['sha256'],len(batch)))
    processed.append(name)
    if len(processed)%25==0:print(json.dumps({'indexed_files':len(processed),'missing_files':len(missing)}),flush=True)

count,hours,geometry,valid_hours,roundtrip_bad,maxU,maxT=db.execute('SELECT COUNT(*),SUM(duration)/3600.,SUM(geometry_ok AND roundtrip_ok),SUM(CASE WHEN geometry_ok AND roundtrip_ok THEN duration ELSE 0 END)/3600.,SUM(NOT roundtrip_ok),MAX(U),MAX(T_upper_estimate) FROM rows').fetchone()
grid=[]
for cer in [.05,.10,.15,.30]:
    for duration in [12.,16.,20.]:
        for mos in [None,2.0,2.5]:
            clause='geometry_ok=1 AND roundtrip_ok=1 AND cer IS NOT NULL AND cer<=? AND duration<=?'
            args=[cer,duration]
            if mos is not None:clause+=' AND mos IS NOT NULL AND mos>=?';args.append(mos)
            n,h=db.execute('SELECT COUNT(*),COALESCE(SUM(duration)/3600.,0) FROM rows WHERE '+clause,args).fetchone()
            grid.append({'max_source_CER':cer,'max_duration':duration,'min_DNSMOS':mos,'rows':n,'hours':h})
duplicates=db.execute('SELECT COUNT(*) FROM (SELECT source_id FROM rows GROUP BY source_id HAVING COUNT(*)>1)').fetchone()[0]
summary={'kind':'ROOT_WorldSpeech_TRAIN_candidate_metadata_token_index_v1','pid':os.getpid(),'converted_revision':REV,
    'processed_files':len(processed),'expected_TRAIN_files':len(files),'missing_files':missing,'all_TRAIN_complete':len(processed)==len(files),
    'rows':count,'hours':hours,'geometry_and_roundtrip_rows':geometry,'geometry_and_roundtrip_hours':valid_hours,
    'roundtrip_bad_rows':roundtrip_bad,'max_target_U':maxU,'max_estimated_T':maxT,'duplicate_source_ID_groups':duplicates,
    'quality_grid_UNSELECTED':grid,'actual_source_label_fixture':record(OUT/'exact_original8199_actual_label_fixture_v1.json'),
    'tokenizer_semantic_sha256':semantic,'helper':record(pathlib.Path(__file__)),
    'geometry_note':'T is conservatively estimated from source duration; actual decoded PCM/header/frame and native memory admission remain required.',
    'source_byte_note':'Staged file size/stat and original downloader SHA record bound; this metadata pass does not rehash all audio-bearing source bytes.',
    'admitted_new_training_hours':0,'normalized_text_duplicates_not_automatically_dropped':True,
    'candidate_filter_selected':False,'source_exclusion_audio_dedup_and_label_quality_checks_complete':False,
    'Torch_imported':'torch' in sys.modules,'GPU_used':False,'audio_decoded':False,'DEV_TEST_read':False}
assert not summary['Torch_imported']
db.close();(OUT/'candidate_metadata_token_index_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['quality_grid_UNSELECTED','missing_files']}),flush=True)
