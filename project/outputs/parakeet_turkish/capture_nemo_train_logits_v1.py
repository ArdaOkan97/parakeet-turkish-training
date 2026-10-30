"""Capture bounded eval joint logits from immutable1871 for an official-loss check.

--prepare-only selects/proves two actual TRAIN clips on CPU. GPU capture needs
--gpu-window-authorized, never constructs an optimizer and writes only NEW files.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoProcessor
from tokenizers import Tokenizer

from runtime import ROOT, OUTPUT, load_model, read_audio
from extend_vocab import extend_processor_and_model, _make_tokenizer
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from train_lora_batched import validate_trainable_checkpoint

RUN = 'adaptive_20261010_continuation_03_resume_B8'
SOURCE_NAMES = ('runtime.py','extend_vocab.py','encoder_lora.py','train_lora_batched.py','batched_lora.py')
TOKENIZER = ROOT/'work/research/extension_roundtrip/tokenizer.json'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def input_sources():
    config_path = OUTPUT/'runs'/RUN/'run_config.json'
    config = read(config_path)
    for name,digest in config['code_sha256'].items():
        require(sha(OUTPUT/name)==digest,'Frozen training source differs: '+name)
    return config, {'config_path':str(config_path),'config_sha256':sha(config_path),
                    'code_sha256':{name:sha(OUTPUT/name) for name in SOURCE_NAMES}}


def choose_rows(pool, processor):
    def candidates(rows,language):
        eligible=[]
        for row in rows:
            if row.get('dataset')!='google/fleurs' or row.get('dataset_split')!='train' or row.get('language')!=language:continue
            ids=processor.tokenizer(row['text'])['input_ids']
            if not (1<=row['encoder_frames']<=128 and 1<=len(ids)<=80 and len(ids)==row['target_tokens']):continue
            if language=='tr' and not any(x>=8193 for x in ids):continue
            if language=='en' and any(x>=8193 for x in ids):continue
            eligible.append((row,ids))
        return sorted(eligible,key=lambda pair:(pair[0]['encoder_frames']*(len(pair[1])+1),pair[0]['source_id']))
    tr=candidates(pool['turkish'],'tr');en=candidates(pool['replay'],'en')
    require(tr and en,'No bounded positive Turkish+English FLEURS TRAIN targets')
    selected=[tr[0],en[0]]
    max_t=max(row['encoder_frames'] for row,ids in selected)
    max_u=max(len(ids) for row,ids in selected)
    require(2*max_t*(max_u+1)*8204<=80_000_000,'Selected full joint tensor exceeds320MB')
    return selected


def verify_rows(selected):
    heldout=[]
    for split in ('validation','test'):
        for line in (ROOT/'work/data'/(split+'.jsonl')).read_text(encoding='utf-8').splitlines():heldout.append(json.loads(line))
    ids={row['source_id'] for row in heldout}
    waves={row['wav_sha256'] for row in heldout}
    sentences={row['sentence_id'] for row in heldout if row.get('sentence_id') is not None}
    for row,labels in selected:
        require(row['dataset_split']=='train' and row['dataset']=='google/fleurs','TRAIN-only capture')
        require(row['source_id'] not in ids and row['wav_sha256'] not in waves and row['sentence_id'] not in sentences,'Held-out overlap')
        require(sha(row['audio_filepath'])==row['wav_sha256'],'TRAIN WAV changed')
        info=sf.info(row['audio_filepath'])
        require(info.samplerate==16000 and info.channels==1 and abs(info.duration-row['duration'])<1e-5,'WAV bounds differ')
        require(all(0<=label<8199 and label!=8192 for label in labels),'Invalid positive label')


def load_evaluation_model(weights, weights_sha256):
    """Shared exact1871 architecture/weights loader; called only in solo GPU tools."""
    expected=ROOT/'work'/(RUN+'_best.pt')
    require(Path(weights).resolve()==expected.resolve() and sha(weights)==weights_sha256,'Exact immutable1871 best checkpoint required')
    config,sources=input_sources()
    base=(ROOT/'work/base_model_path.txt').read_text().strip()
    model,processor=load_model(base)
    vocabulary=extend_processor_and_model(processor,model)
    layout=install_encoder_lora(model,rank=config['rank'],alpha=config['alpha'],last_n_layers=config['last_n_layers'])
    state=torch.load(weights,map_location='cpu',weights_only=True)
    require(state['schema']==2 and state['run_id']==RUN and state['step']==state['selected_step']==1871,'Full1871 source identity')
    validate_trainable_checkpoint(model,state,encoder_lora_metadata(model))
    result=model.load_state_dict(state['weights'],strict=False)
    require(not result.unexpected_keys,'Unexpected source weight key')
    model.eval()
    require(all(not module.training for module in model.modules()),'A model dropout/BN module remains training')
    del state
    return model,processor,{'encoder_lora_metadata':layout,'vocabulary':vocabulary,'sources':sources,
                            'checkpoint_sha256':weights_sha256,'checkpoint_step':1871}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--weights-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--gpu-window-authorized',action='store_true')
    args=parser.parse_args()
    require(not args.output.exists() and args.output.resolve().is_relative_to(ROOT),'NEW workspace output required')
    expected=ROOT/'work'/(RUN+'_best.pt')
    require(args.weights.resolve()==expected.resolve() and sha(args.weights)==args.weights_sha256,'Exact immutable1871 best checkpoint required')
    config,sources=input_sources()
    pool_path=OUTPUT/'runs'/RUN/'training_pool.json'
    pool=read(pool_path)
    base=(ROOT/'work/base_model_path.txt').read_text().strip()
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    processor.tokenizer=_make_tokenizer(processor.tokenizer,Tokenizer.from_file(str(TOKENIZER)))
    require(len(processor.tokenizer)==8199 and processor.tokenizer.convert_tokens_to_ids(processor.blank_token)==8192,'Exact extended tokenizer required')
    selected=choose_rows(pool,processor);verify_rows(selected)
    sources.update(pool_path=str(pool_path),pool_sha256=sha(pool_path),tokenizer_path=str(TOKENIZER),tokenizer_sha256=sha(TOKENIZER))
    preparation={'schema':1,'run_id':RUN,'checkpoint_step':1871,'checkpoint_sha256':args.weights_sha256,
        'producer_sha256':sha(__file__),'sources':sources,'rows':[row for row,labels in selected],
        'labels':[labels for row,labels in selected],'split':'train','model_training':False,'dropout_disabled':True,
        'precision':'FP32 forcedSDPA-MATH, TF32false for deterministic loss-oracle inputs; no productionBF16 forward-equivalence claim'}
    if args.prepare_only:
        require(not torch.cuda.is_initialized(),'CPU preparation initialized CUDA')
        preparation.update(passed=True,cuda_initialized=False,scope='CPU input/source selection only; no logits captured')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(preparation,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({'passed':True,'sources':[row['source_id'] for row,labels in selected],'sizes':[[row['encoder_frames'],len(labels)] for row,labels in selected]}))
        return
    require(args.gpu_window_authorized,'Parent-owned SOLO GPU window required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(6e9/total,device=0)
    torch.manual_seed(20261010)
    with sdpa_kernel(SDPBackend.MATH):
        model,actual_processor,model_evidence=load_evaluation_model(args.weights,args.weights_sha256)
        require(actual_processor.tokenizer._tokenizer.to_str()==processor.tokenizer._tokenizer.to_str(),'Capture tokenizer differs from CPU selection')
        rows=[row for row,labels in selected]
        labels=pad_sequence([torch.tensor(ids,dtype=torch.long) for row,ids in selected],batch_first=True,padding_value=8192)
        target_lengths=torch.tensor([len(ids) for row,ids in selected],dtype=torch.long)
        decoder_ids=torch.cat((torch.full((2,1),8192,dtype=torch.long),labels),dim=1).cuda()
        inputs=dict(actual_processor(audio=[read_audio(row) for row in rows],sampling_rate=16000,return_tensors='pt',padding='longest'))
        mask=inputs['attention_mask'].bool()
        inputs['input_features']=inputs['input_features'].masked_fill(~mask.unsqueeze(-1),0)
        inputs={name:value.cuda() for name,value in inputs.items()}
        with torch.inference_mode():
            encoded=model.encoder(**inputs)
            frame_lengths=encoded.attention_mask.sum(-1).cpu().long()
            require(frame_lengths.tolist()==[row['encoder_frames'] for row in rows],'Current encoder/cache valid lengths differ')
            features=encoded.last_hidden_state[:,:int(frame_lengths.max())]
            ep=model.encoder_projector(features);dp=model.decoder(decoder_ids)
            logits=model.joint(ep[:,:,None],dp[:,None]).float().cpu().contiguous()
        require(logits.shape==(2,int(frame_lengths.max()),int(target_lengths.max())+1,8204) and logits.numel()<=80_000_000,'Actual full logits shape/size differs')
        require(bool(torch.isfinite(logits).all()),'Joint logits nonfinite')
        provenance={**preparation,'source_ids':[row['source_id'] for row in rows],
                    'tokenizer_sha256':sources['tokenizer_sha256'],'model_training':False,'dropout_disabled':True,
                    'encoder_lora_metadata':model_evidence['encoder_lora_metadata'],'vocabulary':model_evidence['vocabulary'],
                    'gpu_max_allocated_bytes':torch.cuda.max_memory_allocated()}
        data={'schema':1,'kind':'actual_fleurs_training_joint_logits','joint_logits':logits,'labels':labels,
              'frame_lengths':frame_lengths,'target_lengths':target_lengths,'blank_id':8192,'durations':[0,1,2,3,4],
              'provenance':provenance}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        torch.save(data,args.output)
    require(sha(args.weights)==args.weights_sha256,'Original checkpoint changed during capture')
    for name,digest in sources['code_sha256'].items():require(sha(OUTPUT/name)==digest,'Source changed during capture')
    require(sha(pool_path)==sources['pool_sha256'] and sha(TOKENIZER)==sources['tokenizer_sha256'],'Inputs changed during capture')
    print(json.dumps({'captured':True,'output':str(args.output),'sha256':sha(args.output),'shape':list(logits.shape),'source_ids':provenance['source_ids']}))


if __name__=='__main__':main()
