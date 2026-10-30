"""Independent bounded read-only verification of completed candidate exports/audio samples."""
import collections, hashlib, json, math, os, pathlib, sqlite3, sys, wave
ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=pathlib.Path(r'C:\Users\ardao\Miniconda3\envs\vlm')
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path[:0]=[str(ROOT/'work/training-env/Lib/site-packages'),str(BASE/'Lib/site-packages'),str(ROOT/'outputs/parakeet_turkish')]
_dll=os.add_dll_directory(str(BASE/'Library/bin'))
from tokenizers import Tokenizer
from metrics import normalize_text

OUT=ROOT/'outputs/parakeet_turkish'
DATA=ROOT/'work/data/WorldSpeech_tr_quality_candidate_v1'
REPORT=OUT/'WorldSpeech_complete_candidate_CPU_independent_verification_v1.json'
REV='f389569345a04eb8ac8fa9405b0d5048ad74b8b3'
TOK='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def rec(path):
    path=pathlib.Path(path).resolve();return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def rows(path):
    with pathlib.Path(path).open(encoding='utf-8') as f:
        for line in f:yield json.loads(line)

assert not REPORT.exists()
prep_path=OUT/'WorldSpeech_TR_TRAIN_audio_candidate_preparation_v1.json'
assert sha(prep_path)=='8aac435e3472d3e304a88c4ac952756f1a01682682ef1d8cc82beedbc694e9e7'
prep=json.loads(prep_path.read_text());assert prep['status']=='complete' and prep['prepared_candidate_clips']==98970
for item in prep['exports']:
    assert rec(item['path'])==item
db=sqlite3.connect((DATA/'preparation.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=FILE')
assert db.execute('SELECT COUNT(*) FROM files').fetchone()[0]==365
assert db.execute('SELECT SUM(selected),SUM(accepted) FROM files').fetchone()==(98972,98970)
assert db.execute('SELECT COUNT(*) FROM rejected').fetchone()[0]==2
for field in ['source_id','embedded_sha','pcm_sha','wav_sha']:
    assert db.execute('SELECT COUNT(DISTINCT '+field+') FROM accepted').fetchone()[0]==98970
for field in ['embedded_sha','pcm_sha','wav_sha']:
    assert db.execute('SELECT COUNT(*) FROM accepted a JOIN forbidden_audio f ON a.'+field+'=f.sha').fetchone()[0]==0
assert db.execute('SELECT COUNT(*) FROM accepted a JOIN forbidden_ids f ON a.source_id=f.source_id').fetchone()[0]==0

samples={x[0] for x in db.execute('SELECT source_id FROM accepted WHERE embedded_sha IN (SELECT MIN(embedded_sha) FROM accepted GROUP BY filename)')}
nonempty_shards=db.execute('SELECT COUNT(DISTINCT filename) FROM accepted').fetchone()[0]
assert nonempty_shards==364 and db.execute('SELECT COUNT(*) FROM files WHERE selected=0 AND accepted=0').fetchone()[0]==1
assert len(samples)==nonempty_shards
samples.update(x[0] for x in db.execute("SELECT source_id FROM accepted ORDER BY json_extract(qc,'$.perclip_joint_bytes') DESC,source_id LIMIT 32"))
t=Tokenizer.from_file(str(OUT/'checkpoint/tokenizer.json'))
semantic=hashlib.sha256(json.dumps(json.loads(t.to_str()),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
assert semantic==TOK and t.get_vocab_size()==8199 and t.token_to_id('<blank>')==8192
formats=collections.Counter();source_counts=collections.Counter();count=frames=tokens=read_pcm_bytes=0;sampled=0
max_joint=0;max_clip=0.;max_peak=0.;min_RMS=math.inf;source_urls=set();sessions=set();wave_stat_count=0
ordered=hashlib.sha256();label_stream=hashlib.sha256()
for row,meta,qc in zip(rows(DATA/'candidate_train.jsonl'),rows(DATA/'candidate_frame_metadata.jsonl'),rows(DATA/'candidate_audio_QC.jsonl'),strict=True):
    assert row['dataset']=='disco-eth/WorldSpeech' and row['dataset_revision']==REV and row['dataset_config']=='tr_tr'
    assert row['dataset_split']==row['training_role']=='train' and row['language']=='tr'
    assert row['source_id']==meta['source_id'] and row['text']==row['reference']
    labels=t.encode(row['text'],add_special_tokens=False).ids
    label_sha=hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()
    assert len(labels)==row['target_tokens']==meta['target_tokens'] and label_sha==row['label_ids_sha256']==meta['label_ids_sha256']
    assert 1<=len(labels)<=180 and 0<=min(labels) and max(labels)<8199 and 8192 not in labels
    normalized=normalize_text(row['text'],'tr')
    assert normalized and normalize_text(t.decode(labels,skip_special_tokens=True),'tr')==normalized==row['reference_normalized']
    normalized_sha=hashlib.sha256(normalized.encode()).hexdigest()
    assert db.execute('SELECT 1 FROM heldout_text WHERE sha=?',(normalized_sha,)).fetchone() is None
    path=pathlib.Path(row['audio_filepath']);assert path.is_relative_to(DATA/'audio')
    stat=path.stat();assert {'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}==meta['canonical_WAV_stat_at_preparation']
    wave_stat_count+=1;n=meta['actual_PCM_frames'];mel=n//160;T=(mel+7)//8;U=len(labels)
    assert n/16000==row['duration'] and .5<=row['duration']<=20.001
    assert meta['valid_mel_frames']==mel and meta['encoder_frames']==T
    joint=T*(U+1)*8204*4;assert joint==qc['perclip_joint_bytes']<=1645131712
    assert qc['source_rate']==24000 and qc['source_channels']==2 and abs(qc['source_duration']-row['source_duration'])<=.002
    assert 0<qc['RMS']<math.inf and 0<=qc['clipping_fraction']<=1 and 0<=qc['zero_fraction']<1
    assert abs(qc['source_CER'])<=.1 and qc['DNSMOS']>=2
    formats[(qc['source_rate'],qc['source_channels'])]+=1
    if row['source_id'] in samples:
        with wave.open(str(path),'rb') as f:
            assert (f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getnframes(),f.getcomptype())==(16000,1,2,n,'NONE')
            raw=f.readframes(n)
        assert len(raw)==n*2 and hashlib.sha256(raw).hexdigest()==row['canonical_pcm_sha256']
        assert sha(path)==row['wav_sha256']
        assert {'size_bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}==meta['canonical_WAV_stat_at_preparation']
        sampled+=1;read_pcm_bytes+=len(raw)
    source_counts[row['dataset_file']]+=1;source_urls.add(row['source_url']);sessions.add(row['session_date'])
    ordered.update(row['source_id'].encode()+b'\n');label_stream.update(label_sha.encode()+b'\n')
    count+=1;frames+=n;tokens+=U;max_joint=max(max_joint,joint);max_clip=max(max_clip,qc['clipping_fraction']);max_peak=max(max_peak,qc['resampled_float_peak']);min_RMS=min(min_RMS,qc['RMS'])
    if count%20000==0:print(json.dumps({'verified_rows':count,'actual_PCM_samples':sampled}),flush=True)
assert count==98970 and frames==23514743540 and tokens==9727079 and len(source_counts)==nonempty_shards and sampled==len(samples)
assert 'torch' not in sys.modules
value={'kind':'ROOT_WorldSpeech_complete_candidate_CPU_independent_verification_v1','passed':True,
 'actual_owned_preparation_session':57812,'actual_owned_preparation_exit_code':0,'actual_owned_preparation_closed_tool_chunk':'791ae7',
 'preparation':rec(prep_path),'verifier':rec(pathlib.Path(__file__)),'exact_tokenizer_semantic_sha256':semantic,
 'exports_rehashed':prep['exports'],'actual_all_record_label_frame_stat_and_heldout_hash_checks':count,
 'actual_WAV_stat_checks':wave_stat_count,'actual_PCM_WAV_hash_reread_clips':sampled,'actual_PCM_payload_bytes_reread':read_pcm_bytes,
 'sample_policy':'One minimum embedded SHA per each of364 nonempty accepted shards plus32 largest joint geometries; all365 source shards scanned, one has no eligible rows. Independent bounded PCM reread; original first416 pilot verification remains separate.',
 'new_candidate_clips_after_exact_audio_exclusion':count,'new_candidate_actual_PCM_hours':frames/16000/3600,
 'source_shards_scanned':365,'nonempty_accepted_source_shards':len(source_counts),'source_URLs':len(source_urls),'session_dates':len(sessions),'source_audio_formats':{str(k):v for k,v in formats.items()},
 'actual_target_tokens':tokens,'mean_target_U':tokens/count,'mean_audio_seconds':frames/16000/count,'ordered_source_ID_stream_sha256':ordered.hexdigest(),
 'ordered_label_SHA_stream_sha256':label_stream.hexdigest(),'max_perclip_joint_bytes':max_joint,'max_PCM_clipping_fraction':max_clip,
 'max_resampled_float_peak':max_peak,'minimum_RMS':min_RMS,'duplicate_source_or_embedded_or_PCM_or_WAV_found':False,
 'exact_existing_TRAIN_replay_audio_overlap_found':False,'fixed_DEV_TEST_ID_audio_normalized_text_overlap_found':False,
 'all_prepared_PCM_independently_reread':False,'all_prepared_PCM_actually_decoded_by_closed_preparer':True,
 'two_duration_mismatch_exclusions':rec(OUT/'WorldSpeech_actual_two_rejections_review_v1.json'),
 'raw_human_labels_preserved':True,'native_batch_resource_fit_proved':False,'independent_TRAIN_ASR_sanity_pending':True,
 'candidate_admitted_to_model':False,'measured_WER_improvement':None,'Torch_imported':False,'GPU_used':False,'TEST_predictions_or_evaluation_used':False,
 'speaker_video_and_perceptual_near_duplicate_disjointness_proven':False,
 'provenance_note':'staging_status.manifest_sha256 hashes the datasets-server acquisition manifest; complete_TRAIN_manifest file hash is separately bound in preparation. They identify different files.'}
REPORT.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value),flush=True)
print(json.dumps(rec(REPORT)),flush=True)
