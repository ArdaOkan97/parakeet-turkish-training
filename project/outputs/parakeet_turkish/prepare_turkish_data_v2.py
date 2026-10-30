"""CPU-only pinned public Turkish expansion into new immutable manifests.

Preserves the entire existing prepared manifest as an exact byte prefix. Adds
eligible original CV train recordings omitted earlier, full FLEURS train, and
CV's separately named validated split. Repeated TRAIN words with distinct audio
are retained. Heldout words, global FLEURS sentence IDs and audio remain excluded.
The new validated source requires an explicit NEW trainer eligibility policy;
the frozen existing trainer/config/sampler are never changed by this helper.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from atomic_io_retry_v1 import write_json
from metrics import normalize_text
from prepare_expanded_turkish import sha256, rows, heldout_filters, forbidden_reason

OUTPUT=Path(__file__).resolve().parent; ROOT=OUTPUT.parent.parent; DATA=ROOT/'work/data'
CV_REPO='ysdede/commonvoice_17_tr_fixed'; CV_REV='9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b'
FL_REPO='google/fleurs'; FL_REV='70bb2e84b976b7e960aa89f1c648e09c59f894dd'
VALIDATED_SHA='3c726f496add3addd0172c7fe6e31373d1215f579b3ddee3674e991842850ac3'
BASE_RUN='adaptive_20261010_continuation_01_resume_B8'


def json_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def pcm_hash(path):
    digest=hashlib.sha256()
    with sf.SoundFile(path) as stream:
        if stream.samplerate!=16000 or stream.channels!=1 or stream.subtype!='PCM_16':
            raise ValueError('Verified mono16kPCM16 required')
        duration=stream.frames/stream.samplerate
        while True:
            block=stream.read(65536,dtype='int16',always_2d=False)
            if not block.size:break
            digest.update(block.astype('<i2',copy=False).tobytes())
    return digest.hexdigest(),duration


def source_metadata(parquet_path,repo,revision,split,file,config):
    parquet=pq.ParquetFile(parquet_path)
    metadata=parquet.read(columns=[name for name in parquet.schema_arrow.names if name!='audio'],use_threads=False).to_pylist()
    result=[]
    for index,item in enumerate(metadata):
        text=item.get('raw_transcription') or item.get('transcription') or item.get('sentence') or item.get('text')
        original=str(item.get('path') or f'{file}#row={index}')
        sentence=item.get('id') if repo==FL_REPO else None
        identity=f'{repo}:{config}:{split}:{sentence}:{original}'
        duration=item.get('duration')
        if duration is None and item.get('num_samples') is not None:duration=item['num_samples']/16000
        result.append({'dataset':repo,'dataset_revision':revision,'dataset_config':config,'dataset_split':split,
            'training_role':'train','dataset_file':file,'source_id':identity,'sentence_id':sentence,
            'original_audio_path':original,'language':'tr','text':text,'reference':text,
            'reference_normalized':normalize_text(text,'tr') if isinstance(text,str) else '',
            'fleurs_transcription':item.get('transcription') if repo==FL_REPO else None,
            'gender':item.get('gender'),'speaker_id':item.get('client_id') or item.get('speaker_id'),
            'source_duration':float(duration) if duration is not None else None,'_row':index})
    return parquet,result


def decode_audio(payload,min_duration,max_duration):
    signal,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
    if not signal.size or not np.isfinite(signal).all():raise ValueError('Empty/nonfinite audio')
    signal=signal.mean(axis=1)
    if rate!=16000:
        divisor=math.gcd(int(rate),16000)
        signal=resample_poly(signal,16000//divisor,int(rate)//divisor)
    duration=len(signal)/16000
    if not min_duration<=duration<=max_duration:return None,duration
    buffer=io.BytesIO();sf.write(buffer,signal,16000,format='WAV',subtype='PCM_16')
    return buffer.getvalue(),duration


def main(args):
    started=time.perf_counter();pa.set_cpu_count(2);pa.set_io_thread_count(2)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    from transformers import AutoTokenizer
    prepared=DATA/f'{args.tag}_prepared.jsonl';accepted=DATA/f'{args.tag}_train.jsonl'
    audit=OUTPUT/f'{args.tag}_preparation.json';labels_path=OUTPUT/f'{args.tag}_tokenizer_acceptance.json'
    if any(path.exists() for path in (prepared,accepted,audit,labels_path)):
        raise FileExistsError('New immutable tag required; no previous data/report overwritten')
    if not args.tag.replace('_','').isalnum():raise ValueError('Safe separate data tag required')
    lock=DATA/'sources.lock.json';lock_data=json.loads(lock.read_text())
    if lock_data[CV_REPO]['revision']!=CV_REV or lock_data[FL_REPO]['revision']!=FL_REV:
        raise ValueError('Pinned existing source revisions differ')
    filters,heldout_paths=heldout_filters(DATA)
    protected=list(DATA.glob('*.jsonl'))+[lock]
    config_path=OUTPUT/'runs'/BASE_RUN/'run_config.json';config=json.loads(config_path.read_text())
    protected.extend(OUTPUT/name for name in config['code_sha256'])
    protected.extend([OUTPUT/'prepare_data.py',OUTPUT/'prepare_expanded_turkish.py'])
    protected_sha={str(path.resolve()):sha256(path) for path in set(protected)}
    base_path=DATA/'tr_expanded_train.jsonl';base_bytes=base_path.read_bytes();base=rows(base_path)
    base_ids={row['source_id'] for row in base}
    if not base_bytes.endswith(b'\n'):raise ValueError('Existing newline-terminated manifest required')
    tokenizer_path=OUTPUT/'checkpoint';tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    tokenizer_sha=json_sha(json.loads(tokenizer._tokenizer.to_str()))
    if tokenizer_sha!=config['config_identity']['extended_tokenizer_sha256']:
        raise ValueError('Exact already-trained Turkish tokenizer differs')
    pool=json.loads((OUTPUT/'runs'/BASE_RUN/'training_pool.json').read_text())
    existing_accepted={row['source_id'] for row in pool['turkish']}
    sources=[('recovered_cv_train',CV_REPO,CV_REV,'train','data/train-00000-of-00001.parquet','default'),
             ('full_fleurs_train',FL_REPO,FL_REV,'train','parquet-data/tr_tr/train-00000-of-00001.parquet','tr_tr'),
             ('cv_validated',CV_REPO,CV_REV,'validated','data/validated-00000-of-00001.parquet','default')]
    source_info=[]
    for name,repo,revision,split,file,configuration in sources:
        path=DATA/'hf_cache'/('datasets--'+repo.replace('/','--'))/'snapshots'/revision/file
        if not path.is_file():raise FileNotFoundError('Pinned cached source absent: '+str(path))
        digest=sha256(path)
        if split=='validated' and digest!=VALIDATED_SHA:raise ValueError('Pinned validated download SHA differs')
        source_info.append({'role':name,'repository':repo,'revision':revision,'split':split,'filename':file,
                            'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size})
    seen_id=set();seen_audio=set();seen_wav=set();seen_pcm=set();speaker_text=set();pcm_by_id={}
    for index,row in enumerate(base):
        if forbidden_reason(row,filters):raise ValueError('Original prepared row overlaps fixedheldout')
        path=Path(row['audio_filepath'])
        if sha256(path)!=row['wav_sha256']:raise ValueError('Original prepared audio changed')
        canonical,duration=pcm_hash(path)
        if abs(duration-row['duration'])>1e-8 or not args.min_duration<=duration<=args.max_duration:
            raise ValueError('Original audio duration/PCM mismatch')
        seen_id.add(row['source_id']);seen_audio.add(row['audio_sha256']);seen_wav.add(row['wav_sha256']);seen_pcm.add(canonical)
        pcm_by_id[row['source_id']]=canonical
        if row.get('speaker_id'):speaker_text.add((row['dataset'],row['speaker_id'],row['reference_normalized']))
        if (index+1)%4096==0:print(f'Existing prepared PCM verified {index+1}/{len(base)}',flush=True)
    # Canonical heldout PCM also catches re-encoded source duplicates.
    heldout_pcm={pcm_hash(Path(row['audio_filepath']))[0] for path in heldout_paths for row in rows(path)}
    additions=[];source_counts={};diagnostics=[]
    audio_dir=DATA/(args.tag+'_audio')/'tr';audio_dir.mkdir(parents=True,exist_ok=True)
    for source in source_info:
        parquet,candidates=source_metadata(Path(source['path']),source['repository'],source['revision'],source['split'],source['filename'],
                                         'tr_tr' if source['repository']==FL_REPO else 'default')
        source.update(rows=parquet.metadata.num_rows,row_groups=parquet.metadata.num_row_groups)
        excludes=Counter();selected={}
        for row in candidates:
            if row['source_id'] in seen_id:excludes['existing_training_source_id']+=1;continue
            if not row['reference_normalized']:excludes['empty_reference']+=1;continue
            reason=forbidden_reason(row,filters)
            if reason:excludes[reason]+=1;continue
            selected[row['_row']]=row
        converted=[];offset=0
        for batch in parquet.iter_batches(batch_size=8,columns=['audio'],use_threads=False):
            for position in range(batch.num_rows):
                row=selected.get(offset+position)
                if row is None:continue
                try:
                    embedded=batch.column(0)[position].as_py();payload=embedded.get('bytes')
                    if not payload:raise ValueError('Missing embedded bytes')
                    embedded_sha=hashlib.sha256(payload).hexdigest()
                    if embedded_sha in filters['source_audio_hashes']:excludes['heldout_source_audio_hash']+=1;continue
                    if embedded_sha in seen_audio:excludes['duplicate_training_embedded_audio']+=1;continue
                    wav,duration=decode_audio(payload,args.min_duration,args.max_duration)
                    if wav is None:excludes['actual_pcm_duration_outside_bounds']+=1;continue
                    wav_sha=hashlib.sha256(wav).hexdigest()
                    samples,_=sf.read(io.BytesIO(wav),dtype='int16',always_2d=False)
                    canonical=hashlib.sha256(samples.astype('<i2',copy=False).tobytes()).hexdigest()
                    if wav_sha in filters['wav_hashes'] or canonical in heldout_pcm:
                        excludes['heldout_canonical_pcm_or_wav_hash']+=1;continue
                    # Write one decoded clip now; never retain full-corpus WAV bytes.
                    name=hashlib.sha256((row['dataset_revision']+':'+row['source_id']).encode()).hexdigest()[:24]+'.wav'
                    path=audio_dir/name
                    if path.exists() and sha256(path)!=wav_sha:raise ValueError('Conflicting newaudio; refusing overwrite')
                    if not path.exists():path.write_bytes(wav)
                    converted.append({**row,'audio_sha256':embedded_sha,'wav_sha256':wav_sha,
                        'canonical_pcm_sha256':canonical,'duration':duration,'sample_rate':16000,
                        'audio_filepath':str(path.resolve())})
                except (ValueError,RuntimeError,sf.LibsndfileError) as error:
                    excludes['invalid_audio']+=1
                    if len(diagnostics)<20:diagnostics.append({'source_id':row['source_id'],'error':str(error)})
            offset+=batch.num_rows
            if offset%512==0:print(f"{source['role']} decoded {offset}/{parquet.metadata.num_rows}; candidates {len(converted)}",flush=True)
        converted.sort(key=lambda row:hashlib.sha256(f"{args.seed}:{source['role']}:{row['source_id']}".encode()).digest())
        appended=[]
        for row in converted:
            if row['audio_sha256'] in seen_audio or row['wav_sha256'] in seen_wav or row['canonical_pcm_sha256'] in seen_pcm:
                excludes['duplicate_training_embedded_or_canonical_pcm']+=1;continue
            speaker=(row['dataset'],row['speaker_id'],row['reference_normalized']) if row.get('speaker_id') else None
            if speaker and speaker in speaker_text:excludes['same_verified_speaker_and_utterance']+=1;continue
            path=Path(row['audio_filepath'])
            if pcm_hash(path)!=(row['canonical_pcm_sha256'],row['duration']):raise ValueError('Exported PCM/hash/duration mismatch')
            output={key:value for key,value in row.items() if not key.startswith('_')}
            output.update(audio_filepath=str(path.resolve()),data_expansion_role=source['role'])
            seen_id.add(row['source_id']);seen_audio.add(row['audio_sha256']);seen_wav.add(row['wav_sha256']);seen_pcm.add(row['canonical_pcm_sha256'])
            if speaker:speaker_text.add(speaker)
            pcm_by_id[row['source_id']]=row['canonical_pcm_sha256'];additions.append(output);appended.append(output)
        source_counts[source['role']]={'appended_clips':len(appended),'appended_hours':math.fsum(row['duration'] for row in appended)/3600,
                                      'exclusions':dict(excludes)}
        del converted,candidates,selected,parquet
        print(json.dumps({'source':source['role'],**source_counts[source['role']]}),flush=True)
    all_rows=base+additions
    if len({row['source_id'] for row in all_rows})!=len(all_rows) or any(forbidden_reason(row,filters) for row in all_rows):
        raise ValueError('Final source/heldout isolation failed')
    accepted_rows=[];label_evidence=[];lexical_excludes=Counter();accepted_sources=Counter();old_accept=set()
    for row in all_rows:
        ids=tokenizer(row['text'])['input_ids']
        reason=None
        if not ids:reason='empty_target'
        elif len(ids)>args.max_tokens:reason='target_exceeds180'
        elif min(ids)<0 or max(ids)>=len(tokenizer) or 8192 in ids:reason='invalid_blank_or_out_of_vocabulary_label'
        elif normalize_text(tokenizer.decode(ids,skip_special_tokens=True),'tr')!=normalize_text(row['text'],'tr'):
            reason='normalized_lexical_roundtrip_mismatch'
        if reason:lexical_excludes[reason]+=1;continue
        accepted_rows.append(row)
        if row['source_id'] in base_ids:old_accept.add(row['source_id'])
        accepted_sources[row.get('data_expansion_role','preserved_existing')]+=1
        label_evidence.append({'source_id':row['source_id'],'target_tokens':len(ids),'target_max_id':max(ids),
                              'label_ids_sha256':json_sha(ids),'canonical_pcm_sha256':pcm_by_id[row['source_id']]})
    if old_accept!=existing_accepted:raise ValueError('Exact original CPU tokenizer acceptance differs from realexistingtrainer')
    if any(sha256(Path(path))!=digest for path,digest in protected_sha.items()):raise ValueError('Protected original data/code changed')
    if any(sha256(Path(source['path']))!=source['sha256'] for source in source_info):raise ValueError('Pinned source changed')
    prepared_temp=Path(str(prepared)+'.private.tmp')
    with prepared_temp.open('wb') as stream:
        stream.write(base_bytes)
        for row in additions:stream.write((json.dumps(row,ensure_ascii=False)+'\n').encode('utf-8'))
    prepared_temp.replace(prepared)
    with accepted.open('x',encoding='utf-8',newline='\n') as stream:
        for row in accepted_rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    counts=Counter(row['reference_normalized'] for row in all_rows)
    result={'schema':1,'kind':'public_turkish_data_v2','status':'prepared_and_CPU_tokenizer_verified; newtrainer_policy_and_acousticcache_pending',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'CPU_only':True,'CUDA_initialized':torch.cuda.is_initialized(),
        'sources':source_info,'source_claims_not_measured':{'cv_validated_published_hours':32.69,'validated_is_separate_named_cleanup_split':True},
        'seed':args.seed,'duration_bounds_seconds':[args.min_duration,args.max_duration],'duration_measurement':'Actual mono16kPCM16 frames',
        'original_prepared_manifest_path':str(base_path.resolve()),'original_prepared_manifest_sha256':sha256(base_path),
        'original_prepared_clips':len(base),'original_CPU_tokenizer_accepted_clips':len(old_accept),
        'original_prepared_records_and_exact_byte_prefix_preserved':prepared.read_bytes()[:len(base_bytes)]==base_bytes,
        'source_additions':source_counts,'prepared_clips':len(all_rows),'prepared_hours':math.fsum(row['duration'] for row in all_rows)/3600,
        'CPU_tokenizer_accepted_clips':len(accepted_rows),'CPU_tokenizer_accepted_hours':math.fsum(row['duration'] for row in accepted_rows)/3600,
        'CPU_tokenizer_accepted_source_counts':dict(accepted_sources),'lexical_exclusions':dict(lexical_excludes),
        'prepared_manifest_path':str(prepared.resolve()),'prepared_manifest_sha256':sha256(prepared),
        'CPU_tokenizer_accepted_manifest_path':str(accepted.resolve()),'CPU_tokenizer_accepted_manifest_sha256':sha256(accepted),
        'tokenizer_semantic_sha256':tokenizer_sha,'max_tokens':args.max_tokens,'vocab_size':len(tokenizer),'blank_id':8192,
        'repeated_training_reference_groups':sum(n>1 for n in counts.values()),'extra_recordings_with_repeated_training_reference':sum(n-1 for n in counts.values() if n>1),
        'training_repeat_policy':'Retain distinct-audio repeated TRAIN words; drop exactsource/embedded/canonicalPCM and proved speaker+utterance duplicates only',
        'heldout_policy':'Exclude allfixedDEVTESTsource IDs/embeddedaudio/WAV/canonicalPCM/globalFLEURSsentence IDs/normalizedreferences; no predictions read',
        'heldout_manifests_sha256':{str(path.resolve()):sha256(path) for path in heldout_paths},'heldout_canonical_pcm_count':len(heldout_pcm),
        'protected_original_files_sha256':protected_sha,'all_protected_original_files_unchanged':True,'invalid_audio_examples':diagnostics,
        'preparation_script_sha256':sha256(Path(__file__)),'elapsed_seconds':time.perf_counter()-started,
        'limitations':['These are CPU tokenizer eligible clips, not actual optimizer exposure or completed acoustic-cache acceptance.',
          'Newtrainer must explicitly allow pinned validated source/training_role=train and repeatedTRAIN words with distinctaudio.',
          'No newmanifest is retrofitted into current exactresume optimizer/sampler/configidentity.',
          'Speaker IDs are absent in these publicsource schemas; gender is never treated as a speakerID.']}
    write_json(labels_path,{'schema':1,'passed':True,'tokenizer_semantic_sha256':tokenizer_sha,'max_tokens':args.max_tokens,
         'existing_actual_trainer_source_set_exact':True,'accepted_source_labels_and_audio':label_evidence})
    write_json(audit,result)
    print(json.dumps({k:result[k] for k in ('status','prepared_clips','prepared_hours','CPU_tokenizer_accepted_clips','CPU_tokenizer_accepted_hours','elapsed_seconds')}),flush=True)


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument('--tag',default='tr_public_v2')
    value.add_argument('--seed',type=int,default=20261010)
    value.add_argument('--max-tokens',type=int,default=180)
    value.add_argument('--min-duration',type=float,default=.5)
    value.add_argument('--max-duration',type=float,default=30.)
    return value


if __name__=='__main__':
    args=parser().parse_args()
    if not 0<args.min_duration<=args.max_duration<=30 or args.max_tokens!=180:raise ValueError('Reviewed bounds/max180required')
    main(args)
