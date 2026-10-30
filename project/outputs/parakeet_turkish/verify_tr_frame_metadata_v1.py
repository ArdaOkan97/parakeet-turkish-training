"""CPU-only Turkish frame metadata proof; no encoder inference or cache writes.

This prepares a future metadata-aware collator. Existing frozen trainers still
require genuine teacher-feature caches and cannot consume this file directly.
All cached tensor shapes are read through mmap, without reading feature values.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
from types import SimpleNamespace

D=Path(__file__).resolve().parent; ROOT=D.parents[1]
MANIFEST_SHA='d2d0f6df539b1e5cfe9488960a5859bc6ab1870a1ffc634b5b707bc2cc2f5704'
LABEL_SHA='42aa606057ba78232ef930f7317675ef318dddb034bc302e4de2247eedd04e3a'
NAMESPACE='3460d398e8e3850b2650'

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def frame_lengths(pcm_frames, *, hop=160, kernel=3, stride=2, factor=8):
    if type(pcm_frames) is not int or pcm_frames<=0:raise ValueError('Positive actual integer PCM length required')
    if (hop,kernel,stride,factor)!=(160,3,2,8):raise ValueError('Only pinned base model length policy supported')
    mel=pcm_frames//hop
    output=mel
    for _ in range(3):output=(output-1)//stride+1
    if mel<1 or output<1:raise ValueError('Positive valid lengths required')
    return mel,output

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    return p

def main(args):
    started=time.perf_counter()
    for path in [args.report,args.metadata]:
        if not path.resolve().is_relative_to(ROOT) or path.exists():raise ValueError('Distinct NEW workspace outputs required')
    if args.report.resolve()==args.metadata.resolve():raise ValueError('Distinct outputs required')
    manifest=ROOT/'work/data/tr_public_v3_train.jsonl';label_proof=D/'tr_public_v3_tokenizer_acceptance.json'
    if sha(manifest)!=MANIFEST_SHA or sha(label_proof)!=LABEL_SHA:raise ValueError('Exact vetted TRAIN inputs required')
    import torch, numpy as np, soundfile as sf
    torch.set_num_threads(2)
    from transformers import AutoFeatureExtractor
    from transformers.models.parakeet.modeling_parakeet import ParakeetPreTrainedModel
    base_path=Path((ROOT/'work/base_model_path.txt').read_text(encoding='utf-8').strip())
    config=json.loads((base_path/'config.json').read_text(encoding='utf-8'))['encoder_config']
    if (config['subsampling_conv_kernel_size'],config['subsampling_conv_stride'],config['subsampling_factor'])!=(3,2,8):raise ValueError('Pinned subsampling changed')
    feature_extractor=AutoFeatureExtractor.from_pretrained(base_path,local_files_only=True)
    if (feature_extractor.hop_length,feature_extractor.n_fft,feature_extractor.sampling_rate)!=(160,512,16000):raise ValueError('Pinned feature extraction changed')
    fake=SimpleNamespace(config=SimpleNamespace(**config))
    official_length=ParakeetPreTrainedModel._get_subsampling_output_length
    synthetic=[]
    for length in [8000,8001,8159,8160,8161,10239,10240,10241,16000,16159,16160,16161,31999,32000,32001,479999,480000]:
        features=feature_extractor([np.zeros(length,dtype=np.float32)],sampling_rate=16000,return_tensors='pt')
        mel,frames=frame_lengths(length)
        actual_mel=int(features['attention_mask'].sum())
        official_frames=int(official_length(fake,torch.tensor([actual_mel],dtype=torch.long))[0])
        if (mel,frames)!=(actual_mel,official_frames):raise ValueError('Actual feature extractor/official subsampling mismatch')
        synthetic.append({'actual_PCM_frames':length,'valid_mel_frames':mel,'valid_encoder_frames':frames})
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line]
    label_data=json.loads(label_proof.read_text(encoding='utf-8'))
    labels={row['source_id']:row for row in label_data['accepted_source_labels_and_audio']}
    if len(rows)!=74883 or len(labels)!=len(rows):raise ValueError('Complete exact TRAIN label map required')
    # Snapshot existing paths, then no file is ever opened for writing here.
    cache_dir=ROOT/'work/encoder-cache'/NAMESPACE
    existing={row['source_id']:cache_dir/(hashlib.sha256(row['source_id'].encode()).hexdigest()+'.pt') for row in rows}
    existing={source:path for source,path in existing.items() if path.exists()}
    args.metadata.parent.mkdir(parents=True,exist_ok=True)
    cached_checks=0;unencoded=0;total_seconds=0.;encoder_frames_total=0
    with args.metadata.open('x',encoding='utf-8',newline='\n') as stream:
        for index,row in enumerate(rows):
            if row['language']!='tr':raise ValueError('TRAIN Turkish rows only')
            info=sf.info(row['audio_filepath'])
            if info.channels!=1 or info.samplerate!=16000 or info.frames/info.samplerate!=row['duration']:raise ValueError('Actual PCM header differs from vetted data')
            mel,frames=frame_lengths(int(info.frames));target=labels[row['source_id']]
            if row['source_id'] in existing:
                item=torch.load(existing[row['source_id']],map_location='cpu',weights_only=True,mmap=True)
                if (item['source_id']!=row['source_id'] or item['features'].ndim!=2 or item['features'].shape!=(frames,1024)
                    or item['labels'].numel()!=target['target_tokens']
                    or hashlib.sha256(json.dumps(item['labels'].tolist(),sort_keys=True).encode()).hexdigest()!=target['label_ids_sha256']):
                    raise ValueError('Existing original cached frame/label metadata differs:'+row['source_id'])
                cached_checks+=1;del item
            else:unencoded+=1
            metadata={'source_id':row['source_id'],'actual_PCM_frames':int(info.frames),'valid_mel_frames':mel,
                'encoder_frames':frames,'target_tokens':target['target_tokens'],'label_ids_sha256':target['label_ids_sha256'],
                'teacher_feature_values_required_for_TR_preservation':False}
            stream.write(json.dumps(metadata,ensure_ascii=False)+'\n')
            total_seconds+=row['duration'];encoder_frames_total+=frames
            if (index+1)%10000==0:print('CPU frame metadata '+str(index+1)+'/'+str(len(rows)),flush=True)
    model_file=Path(official_length.__code__.co_filename);extractor_file=Path(feature_extractor.__call__.__code__.co_filename)
    report={'schema':1,'kind':'TR_metadata_only_future_cache_CPU_verification_v1','passed':True,'CPU_only':True,
        'CUDA_initialized':torch.cuda.is_initialized(),'helper_sha256':sha(__file__),
        'manifest_sha256':MANIFEST_SHA,'labels_proof_sha256':LABEL_SHA,'metadata_path':str(args.metadata.resolve()),'metadata_sha256':sha(args.metadata),
        'clips':len(rows),'actual_PCM_hours':total_seconds/3600,'encoder_frames_total':encoder_frames_total,
        'existing_cache_snapshot_rows':len(existing),'all_existing_cache_frame_label_checks':cached_checks,'uncached_rows_without_encoder_inference':unencoded,
        'synthetic_actual_HF_feature_length_checks':synthetic,'formula':'mel=floor(actualPCMframes/160); encoder=ceil(mel/8)',
        'official_feature_extractor_source_sha256':sha(extractor_file),'official_encoder_source_sha256':sha(model_file),
        'base_config_sha256':sha(base_path/'config.json'),'cache_files_written':0,'feature_tensor_values_read':False,
        'WAV_content_rehashed_in_this_audit':False,'WAV_header_frames_checked':len(rows),
        'required_future_changes':['New versioned trainer uses CPU TR frame/target metadata, genuine replay teacher features only.',
            'New versioned collator must not materialize or require TR teacher feature tensors; preserve replay feature bytes and preservation index semantics.',
            'Root-owned bounded GPU check validates actual masks/frame lengths on diverse new TRAIN audio before future adoption.'],
        'can_current_frozen_trainer_consume_metadata':False,'quality_or_numeric_equivalence_claimed':False,
        'elapsed_seconds':time.perf_counter()-started}
    if sha(manifest)!=MANIFEST_SHA or sha(label_proof)!=LABEL_SHA:raise ValueError('Frozen inputs changed during audit')
    if torch.cuda.is_initialized():raise ValueError('CPU-only audit unexpectedly initialized CUDA')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'clips':len(rows),'existing_cache_checks':cached_checks,'uncached_rows':unencoded,'elapsed_seconds':report['elapsed_seconds']}))

if __name__=='__main__':main(parser().parse_args())
