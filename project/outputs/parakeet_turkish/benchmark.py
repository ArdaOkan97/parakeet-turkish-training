"""Fixed FLEURS WER/CER and actual-path word-confidence evaluation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from runtime import DATA, OUTPUT, load_model, inputs_for
from confidence import generate_with_confidence, align_word_confidences, confidence_report, fit_temperature
from metrics import score_transcriptions

LANGUAGES = ['en', 'es', 'it', 'fr', 'de', 'tr']

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode('utf-8')).hexdigest()

def model_fingerprint(model, processor):
    """Hash all weights/buffers one tensor at a time, including frozen encoder.

    This also works for bf16 tensors: hash their byte view instead of converting
    values to another floating-point dtype. Memory overhead is bounded by the
    largest tensor, not the complete model. No extra model forward pass occurs.
    """
    state_hash=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        state_hash.update(name.encode('utf-8')+b'\0')
        if torch.is_tensor(value):
            state_hash.update(str(value.dtype).encode('ascii')+b'\0')
            state_hash.update(json.dumps(list(value.shape)).encode('ascii')+b'\0')
            byte_view=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            state_hash.update(memoryview(byte_view))
            del byte_view
        else:
            state_hash.update(json.dumps(value,sort_keys=True,default=str).encode('utf-8'))
    tokenizer=processor.tokenizer
    backend=getattr(tokenizer,'backend_tokenizer',None)
    tokenizer_state={
        'backend':backend.to_str() if backend is not None else tokenizer.get_vocab(),
        'special_tokens_map':getattr(tokenizer,'special_tokens_map',{}),
        'padding_side':getattr(tokenizer,'padding_side',None),
        'truncation_side':getattr(tokenizer,'truncation_side',None),
    }
    processor_state={}
    for name in getattr(processor,'attributes',[]):
        component=getattr(processor,name,None)
        if name!='tokenizer' and component is not None and hasattr(component,'to_dict'):
            processor_state[name]=component.to_dict()
    config=model.config.to_dict() if hasattr(model.config,'to_dict') else vars(model.config)
    generation=getattr(model,'generation_config',None)
    generation=generation.to_dict() if generation is not None and hasattr(generation,'to_dict') else None
    code_hashes={}
    for filename in ('benchmark.py','metrics.py','confidence.py','runtime.py'):
        source=Path(__file__).resolve().parent/filename
        if source.exists():code_hashes[filename]=hashlib.sha256(source.read_bytes()).hexdigest()
    identity={
        'model_state_sha256':state_hash.hexdigest(),
        'model_config_sha256':_json_hash(config),
        'generation_config_sha256':_json_hash(generation),
        'tokenizer_sha256':_json_hash(tokenizer_state),
        'processor_sha256':_json_hash(processor_state),
        'evaluation_code_sha256':code_hashes,
        'inference_dtype':'CUDA bf16 autocast',
    }
    return {'sha256':_json_hash(identity),'identity':identity}

def prepare_evaluation_metadata(model,processor,folder,languages,splits,limit):
    """Refuse prediction reuse when model, code, tokenizer or input differs."""
    if limit<0:raise ValueError('Evaluation limit cannot be negative')
    fingerprint=model_fingerprint(model,processor)
    metadata_path=folder/'metadata.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('schema_version')!=1 or metadata.get('model_fingerprint')!=fingerprint:
            raise RuntimeError(f'Stale evaluation cache in {folder}: model/tokenizer/config/code fingerprint changed. Use a fresh result tag.')
    else:
        if any(folder.glob('*.jsonl')) or (folder/'summary.json').exists():
            raise RuntimeError(f'Unverified existing evaluation results in {folder}: metadata.json is absent. Use a fresh result tag.')
        metadata={'schema_version':1,'model_fingerprint':fingerprint,'manifests':{}}
    selected={}
    for language in languages:
        for split in splits:
            key=f'{split}_{language}'
            manifest=DATA/f'{key}.jsonl'
            rows=read_jsonl(manifest)
            if limit:rows=rows[:limit]
            identities=[row['source_id'] for row in rows]
            if len(set(identities))!=len(identities):
                raise RuntimeError(f'Duplicate evaluation source IDs in {manifest}')
            if not rows:raise RuntimeError(f'Empty evaluation manifest: {manifest}')
            manifest_state={
                'path':str(manifest.resolve()),
                'file_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'source_ids':identities,
                'utterances':len(rows),
                'limit':limit,
            }
            previous=metadata['manifests'].get(key)
            if previous is not None and previous!=manifest_state:
                raise RuntimeError(f'Stale evaluation cache for {key} in {folder}: manifest, source IDs, or limit changed. Use a fresh result tag.')
            if previous is None and (folder/f'{key}.jsonl').exists():
                raise RuntimeError(f'Unverified existing prediction file for {key} in {folder}. Use a fresh result tag.')
            metadata['manifests'][key]=manifest_state
            selected[key]=rows
    write_json(metadata_path,metadata)
    return selected

def evaluate(model, processor, tag, languages=LANGUAGES, splits=('validation','test'), limit=0):
    model.eval()
    folder=OUTPUT/'results'/tag; folder.mkdir(parents=True,exist_ok=True)
    samples_by_key=prepare_evaluation_metadata(model,processor,folder,languages,splits,limit)
    summary={}
    for language in languages:
        all_rows={}
        for split in splits:
            samples=samples_by_key[f'{split}_{language}']
            path=folder/f'{split}_{language}.jsonl'
            previous=read_jsonl(path) if path.exists() else []
            by_id={r['source_id']:r for r in previous}
            if len(by_id)!=len(previous):
                raise RuntimeError(f'Duplicate cached prediction source IDs in {path}')
            expected={row['source_id']:row for row in samples}
            if set(by_id)-set(expected):
                raise RuntimeError(f'Cached predictions outside the current evaluation sample set in {path}')
            for source_id,result in by_id.items():
                row=expected[source_id]
                if result.get('reference')!=row['text'] or result.get('language')!=language or result.get('split')!=split or result.get('sentence_id')!=row.get('sentence_id'):
                    raise RuntimeError(f'Cached prediction provenance mismatch for {source_id} in {path}')
            rows=[]; start=time.perf_counter()
            with path.open('a',encoding='utf-8',newline='\n') as stream:
                for index,row in enumerate(samples):
                    if row['source_id'] in by_id:
                        result=by_id[row['source_id']]
                    else:
                        inputs=inputs_for(processor,row)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            generated,conf=generate_with_confidence(model,processor.tokenizer,inputs,language=language)
                        hypothesis=processor.decode(generated.sequences[0],skip_special_tokens=True)
                        if hypothesis.strip()!=conf['text'].strip():
                            raise RuntimeError(f'Confidence path mismatch: {hypothesis!r} vs {conf["text"]!r}')
                        result={'source_id':row['source_id'],'sentence_id':row.get('sentence_id'),
                                'language':language,'split':split,'reference':row['text'],
                                'dataset_revision':row.get('dataset_revision'),
                                'audio_sha256':row.get('audio_sha256'),'wav_sha256':row.get('wav_sha256'),
                                'duration':row.get('duration'),
                                'hypothesis':hypothesis,'confidence':conf,
                                'alignment':align_word_confidences(row['text'],conf,language)}
                        stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                    rows.append(result)
                    if (index+1)%10==0 or index+1==len(samples):
                        print(f'{tag} {split} {language}: {index+1}/{len(samples)} {time.perf_counter()-start:.1f}s',flush=True)
            all_rows[split]=rows
            metrics=score_transcriptions(rows,language=language)
            metrics.pop('details')
            summary[f'{split}_{language}']={'asr':metrics,
                 'raw_confidence':confidence_report([r['alignment'] for r in rows])}
        if 'validation' in all_rows and all_rows['validation']:
            fitted=fit_temperature([r['alignment'] for r in all_rows['validation']])
            summary[f'calibration_{language}']=fitted
            if 'test' in all_rows:
                summary[f'test_{language}']['calibrated_confidence']=confidence_report(
                    [r['alignment'] for r in all_rows['test']],fitted['temperature'])
        write_json(folder/'summary.json',summary)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path');parser.add_argument('--tag',required=True)
    parser.add_argument('--languages',nargs='+',default=LANGUAGES)
    parser.add_argument('--splits',nargs='+',default=['validation','test'])
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    model,processor=load_model(args.model_path)
    evaluate(model,processor,args.tag,args.languages,args.splits,args.limit)
