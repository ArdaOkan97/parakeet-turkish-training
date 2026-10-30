import hashlib, io, json, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from transformers import AutoTokenizer

ROOT=Path.cwd(); OUT=ROOT/'outputs/parakeet_turkish'; DATA=ROOT/'work/data'
sys.path.insert(0,str(OUT))
from metrics import normalize_text
from prepare_expanded_turkish import sha256, rows, heldout_filters, forbidden_reason

start=time.perf_counter()
prep=json.loads((OUT/'expanded_turkish_preparation.json').read_text(encoding='utf-8'))
manifest=DATA/'tr_expanded_train.jsonl'; records=rows(manifest)
assert sha256(manifest)==prep['manifest_sha256']
filters,paths=heldout_filters(DATA)
sources=set(); refs=set(); source_hashes=set(); wav_hashes=set()
for record in records:
 assert record['language']=='tr' and record['dataset_split']=='train'
 assert not forbidden_reason(record,filters)
 assert record['reference_normalized']==normalize_text(record['reference'],'tr')
 assert record['source_id'] not in sources and record['reference_normalized'] not in refs
 assert record['audio_sha256'] not in source_hashes and record['wav_sha256'] not in wav_hashes
 sources.add(record['source_id']);refs.add(record['reference_normalized']);source_hashes.add(record['audio_sha256']);wav_hashes.add(record['wav_sha256'])
 wav=Path(record['audio_filepath']);assert sha256(wav)==record['wav_sha256']
 signal,sr=sf.read(wav,dtype='float32',always_2d=True);info=sf.info(wav)
 assert signal.shape[1]==1 and sr==16000 and info.subtype=='PCM_16' and np.isfinite(signal).all()
 assert .5<=len(signal)/sr<=30 and abs(record['duration']-len(signal)/sr)<1e-9
parquet=DATA/'hf_cache/datasets--ysdede--commonvoice_17_tr_fixed/snapshots/9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b/data/train-00000-of-00001.parquet'
assert sha256(parquet)==prep['source_parquet_sha256']
pf=pq.ParquetFile(parquet)
meta=pf.read(columns=['transcription']).column(0).to_pylist()
cv={int(r['original_audio_path'].rsplit('#row=',1)[1]):r for r in records if r['dataset']=='ysdede/commonvoice_17_tr_fixed'}
for index,record in cv.items():
 assert record['reference']==meta[index]
 assert record['source_id']==f"ysdede/commonvoice_17_tr_fixed:default:train:None:data/train-00000-of-00001.parquet#row={index}"
offset=0
for batch in pf.iter_batches(batch_size=512,columns=['audio']):
 for i in range(batch.num_rows):
  record=cv.get(offset+i)
  if record:assert hashlib.sha256(batch.column(0)[i].as_py()['bytes']).hexdigest()==record['audio_sha256']
 offset+=batch.num_rows
original_fleurs={r['source_id']:r for r in rows(DATA/'tr_fleurs_train.jsonl')}
for record in records:
 if record['dataset']=='google/fleurs':
  for key in ('reference','sentence_id','dataset_revision','audio_sha256','wav_sha256'):
   assert record[key]==original_fleurs[record['source_id']][key]
unchanged={name:sha256(ROOT/name)==wanted for name,wanted in prep['protected_original_files_sha256'].items()}
assert all(unchanged.values())
tokenizer=AutoTokenizer.from_pretrained(str((OUT/'checkpoint').resolve()),local_files_only=True)
semantic=hashlib.sha256(json.dumps(json.loads(tokenizer._tokenizer.to_str()),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
audit=json.loads((OUT/'lora_training_data_verification.json').read_text(encoding='utf-8'))
assert semantic==audit['extended_tokenizer_semantic_sha256'],(semantic,audit['extended_tokenizer_semantic_sha256'])
counts=Counter();accepted=[];examples=[];tokens=Counter();per_source={}
for record in records:
 ids=tokenizer(record['text'])['input_ids']; reasons=[]
 if not ids:reasons.append('empty_target')
 if len(ids)>180:reasons.append('over180_target_tokens')
 if any(x<0 or x>=8199 or x==8192 for x in ids):reasons.append('invalid_vocab_or_blank')
 decoded=tokenizer.decode(ids,skip_special_tokens=True)
 if normalize_text(decoded,'tr')!=normalize_text(record['text'],'tr'):reasons.append('lexical_roundtrip')
 if reasons:
  counts.update(reasons)
  if len(examples)<20:examples.append({'source_id':record['source_id'],'reasons':reasons,'tokens':len(ids)})
 else:
  accepted.append(record);tokens.update(ids)
  group=per_source.setdefault(record['dataset'],{'clips':0,'hours':0,'target_tokens':0})
  group['clips']+=1;group['hours']+=record['duration']/3600;group['target_tokens']+=len(ids)
report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'device':'CPU only; no GPU or model loaded','scope':'Recomputed source references/source audio bytes, all prepared WAV bytes/finite signals/actual lengths, all held-out isolation and protected hashes; exact published six-character HF tokenizer filter at max180 targets',
 'prepared_clips':len(records),'prepared_hours':sum(r['duration'] for r in records)/3600,'manifest_sha256':sha256(manifest),'source_parquet_sha256':sha256(parquet),'protected_original_files_unchanged':unchanged,
 'lexical_filter':{'tokenizer_semantic_sha256':semantic,'equals_audited_training_tokenizer':True,'vocab_size':len(tokenizer),'blank_id':8192,'max_tokens':180,'accepted_clips':len(accepted),'accepted_hours':sum(r['duration'] for r in accepted)/3600,'by_source':per_source,'excluded_clips':len(records)-len(accepted),'exclusion_counts_nonexclusive':dict(counts),'examples':examples,'new_character_label_occurrences':{str(i):tokens[i] for i in range(8193,8199)},'status':'Exact tokenizer filter verified on CPU; trainer must still apply these checks and verify acoustic/cache acceptance before training'},
 'heldout_source_ids':len(filters['source_ids']),'heldout_global_sentence_ids':len(filters['sentence_ids']),'heldout_predictions_read':False,'audit_script_sha256':sha256(Path(__file__)),'elapsed_seconds':time.perf_counter()-start}
(OUT/'expanded_turkish_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ('passed','prepared_clips','prepared_hours','lexical_filter','elapsed_seconds')},ensure_ascii=False))
