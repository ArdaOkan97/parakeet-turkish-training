"""TR-only missing frozen-base feature cache; existing/replay bytes stay untouched.

Modes: plan is CPU-only; smoke compares actual masked frame/label metadata against
singleton encoding without writing caches; populate requires that exact smoke.
TR feature preservation terms are zero in frozen batched_lora.py. Batch-shaped
BF16 feature differences are reported rather than called numerical equivalence.
Root must own the sole GPU window. This cannot attach to an active trainer.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from atomic_io_retry_v1 import write_json, commit_writer
OUTPUT=Path(__file__).resolve().parent;ROOT=OUTPUT.parents[1]
BASE_REVISION='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
MANIFEST_SHA='d2d0f6df539b1e5cfe9488960a5859bc6ab1870a1ffc634b5b707bc2cc2f5704'
TOKENIZER_SHA='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
NAMESPACE='3460d398e8e3850b2650'
PINS={'runtime.py':'854380ca0938c2dc54e1a8c0ad703a837521f22c7f3928121cf4f4ae0e2aaa6c',
      'batched_lora.py':'d8fb768eb34414f37deddd1508544da4b1a3d41f663565489138a7e662e72c1e',
      'atomic_io_retry_v1.py':'2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0',
      'resource_watchdog_v2.py':'64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641',
      'paired_windows_commit_telemetry_v1.py':'2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1024*1024),b''):digest.update(value)
    return digest.hexdigest()


def cache_path(row):return ROOT/'work/encoder-cache'/NAMESPACE/(hashlib.sha256(row['source_id'].encode()).hexdigest()+'.pt')


def pad_singleton_inputs(items):
    """Keep valid singleton mel values exact, adding only right-side zero padding."""
    import torch
    if not items:raise ValueError('Positive batch required')
    if any(set(item)!={'input_features','attention_mask'} for item in items):raise ValueError('Reviewed encoder input keys required')
    width=items[0]['input_features'].shape[-1];length=max(item['input_features'].shape[1] for item in items)
    features=items[0]['input_features'].new_zeros((len(items),length,width))
    masks=items[0]['attention_mask'].new_zeros((len(items),length))
    for index,item in enumerate(items):
        value,mask=item['input_features'],item['attention_mask']
        if value.ndim!=3 or value.shape[0]!=1 or value.shape[-1]!=width or mask.shape!=(1,value.shape[1]):raise ValueError('Singleton mel/mask dimensions differ')
        valid=int(mask.sum());expected=torch.arange(mask.shape[1])<valid
        if valid<=0 or not torch.equal(mask[0].bool(),expected):raise ValueError('Positive right-padded singleton mask required')
        features[index,:value.shape[1]]=value[0];masks[index,:mask.shape[1]]=mask[0]
    features=features.masked_fill(~masks.bool().unsqueeze(-1),0)
    return {'input_features':features,'attention_mask':masks}


def smoke_rows(rows,count=16):
    ordered=sorted(rows,key=lambda row:(row['duration'],row['source_id']))
    if len(ordered)<count:raise ValueError('Enough missing TRAIN rows required for smoke')
    return [ordered[round(i*(len(ordered)-1)/(count-1))] for i in range(count)]


def inputs_cpu(processor,row):
    import soundfile as sf
    if sha(row['audio_filepath'])!=row['wav_sha256']:raise ValueError('Training WAV changed:'+row['source_id'])
    audio,sr=sf.read(row['audio_filepath'],dtype='float32')
    if sr!=16000 or audio.ndim!=1 or len(audio)/sr!=row['duration']:raise ValueError('Actual mono16k duration differs')
    return dict(processor(audio=[audio],sampling_rate=16000,return_tensors='pt'))


def encode(model,items):
    import torch
    inputs=pad_singleton_inputs(items)
    expected=model.encoder._get_subsampling_output_length(inputs['attention_mask'].sum(-1)).tolist()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model.encoder(**{key:value.cuda() for key,value in inputs.items()})
    actual=output.attention_mask.sum(-1).cpu().tolist()
    if actual!=expected:raise ValueError('Batched output mask/valid frame lengths differ')
    features=[output.last_hidden_state[index,:length].cpu().contiguous() for index,length in enumerate(actual)]
    if any(not torch.isfinite(value).all() or value.ndim!=2 or value.shape[0]!=length for value,length in zip(features,actual)):raise ValueError('Finite valid cache feature dimensions required')
    return features,actual


def main(args):
    started=time.perf_counter();report=args.report.resolve()
    if not report.is_relative_to(ROOT) or report.exists():raise ValueError('NEW workspace report required')
    if args.batch_size!=8:raise ValueError('Reviewed physical batch8 required')
    if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0):raise ValueError('Positive finite wall budget required')
    if args.stop_file and (not args.stop_file.resolve().is_relative_to(ROOT) or args.stop_file.exists()):raise ValueError('NEW owned workspace stop path required')
    if args.manifest.resolve()!=(ROOT/'work/data/tr_public_v3_train.jsonl').resolve() or sha(args.manifest)!=MANIFEST_SHA:raise ValueError('Exact vetted v3 TRAIN manifest required')
    sources={name:sha(OUTPUT/name) for name in PINS}
    for name,expected in PINS.items():
        if expected is not None and sources[name]!=expected:raise ValueError('Frozen implementation changed:'+name)
    rows=[json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line]
    if len(rows)!=74883 or any(row['language']!='tr' for row in rows):raise ValueError('Complete vetted TR-only pool required')
    missing=[row for row in rows if not cache_path(row).exists()]
    value={'schema':1,'kind':'TR_only_batched_frozen_base_cache_v1','mode':args.mode,'status':'planned_not_run',
        'helper_sha256':sha(__file__),'manifest_sha256':MANIFEST_SHA,'tokenizer_semantic_sha256':TOKENIZER_SHA,
        'base_revision':BASE_REVISION,'cache_namespace':NAMESPACE,'source_sha256':sources,'batch_size':8,
        'TR_rows':len(rows),'missing_TR_cache_rows':len(missing),'missing_TR_actual_audio_hours':math.fsum(row['duration'] for row in missing)/3600,
        'existing_TR_cache_rows':len(rows)-len(missing),'replay_rows_processed':0,'existing_cache_files_opened_for_writing':0,
        'TR_preservation_terms_zero':True,'cross_shape_BF16_equivalence_claimed':False,'old_and_replay_cache_policy':'Write missingTR destinations only; preserve every existing destination',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'GPU_initialized':False}
    if args.mode=='plan':write_json(report,value);print(json.dumps(value));return
    if not args.root_owned_solo_gpu:raise ValueError('Root-owned soleGPU window must be explicitly asserted')
    if args.mode=='populate':
        proof=json.loads(args.validated_smoke.read_text(encoding='utf-8')) if args.validated_smoke else {}
        if not(proof.get('passed') is True and proof.get('mode')=='smoke' and proof.get('helper_sha256')==sha(__file__) and proof.get('manifest_sha256')==MANIFEST_SHA and proof.get('mask_and_label_metadata_parity_passed') is True):raise ValueError('Exacthelper actualGPU metadata smoke required')
        value['validated_smoke_sha256']=sha(args.validated_smoke)
    import torch
    from runtime import load_model
    from transformers import AutoTokenizer
    from resource_watchdog_v2 import sample_memory, ResourceGuard, worker_process_commit_reason
    import os
    total=torch.cuda.get_device_properties(0).total_memory
    if not 15e9<=total<=18e9:raise ValueError('Reviewed16GB-class device required')
    torch.cuda.set_per_process_memory_fraction(9e9/total,device=0)
    model,processor=load_model();model.eval().requires_grad_(False)
    processor.tokenizer=AutoTokenizer.from_pretrained(OUTPUT/'checkpoint',local_files_only=True)
    tokenizer_sha=hashlib.sha256(json.dumps(json.loads(processor.tokenizer._tokenizer.to_str()),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    namespace=hashlib.sha256((BASE_REVISION+processor.tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
    if tokenizer_sha!=TOKENIZER_SHA or namespace!=NAMESPACE:raise ValueError('Unchanged exact label IDs/cache namespace required')
    base_path=(ROOT/'work/base_model_path.txt').read_text().strip()
    if BASE_REVISION not in base_path:raise ValueError('Pinned originalbase encoder required')
    selected=smoke_rows(missing) if args.mode=='smoke' else sorted(missing,key=lambda row:(row['duration'],row['source_id']))
    protect=sorted([cache_path(row) for row in rows if cache_path(row).exists()])[:16]
    replay=[json.loads(line) for line in (ROOT/'work/data/replay_train.jsonl').read_text(encoding='utf-8').splitlines() if line][:16]
    protect+= [cache_path(row) for row in replay if cache_path(row).exists()]
    protected={str(path):sha(path) for path in protect};resource=ResourceGuard();max_allocated=0;max_reserved=0;written=[];smoke=[];stop_reason=None
    value.update(GPU_initialized=True,status='running',allocator_cap_bytes=9_000_000_000,cleanup_threshold_bytes=4_000_000_000,
        root_asserted_solo_gpu=True,protected_bounded_existing_files_sha256=protected,existing_caches_exhaustively_rehashed=False)
    write_json(report,value)
    for offset in range(0,len(selected),8):
        group=selected[offset:offset+8];items=[inputs_cpu(processor,row) for row in group]
        features,lengths=encode(model,items)
        for row,item,feature,length in zip(group,items,features,lengths):
            labels=processor.tokenizer(row['text'],return_tensors='pt')['input_ids'][0].cpu()
            if not 1<=len(labels)<=180 or int(labels.max())>=8199 or 8192 in labels:raise ValueError('Exactvetted targetbounds required')
            if args.mode=='smoke':
                single,single_length=encode(model,[item])
                if single_length!=[length] or single[0].shape!=feature.shape:raise ValueError('Actualsingleton/batched validframes differ')
                difference=(single[0].float()-feature.float()).double();norm=single[0].double().norm().item()
                smoke.append({'source_id':row['source_id'],'valid_encoder_frames':length,'target_tokens':len(labels),
                    'label_ids_sha256':hashlib.sha256(json.dumps(labels.tolist()).encode()).hexdigest(),
                    'valid_input_mask_frames':int(item['attention_mask'].sum()),
                    'feature_relative_L2_difference':difference.norm().item()/max(norm,1e-30),'feature_max_abs_difference':difference.abs().max().item()})
            else:
                destination=cache_path(row)
                if destination.exists():raise FileExistsError('Existingcache neveroverwritten:'+str(destination))
                cache_item={'features':feature,'labels':labels,'source_id':row['source_id']}
                commit_writer(destination,lambda temporary,item=cache_item:torch.save(item,temporary))
                written.append({'source_id':row['source_id'],'path':str(destination),'sha256':sha(destination),'valid_encoder_frames':length,'target_tokens':len(labels)})
        del features,items
        torch.cuda.synchronize();max_allocated=max(max_allocated,torch.cuda.max_memory_allocated());max_reserved=max(max_reserved,torch.cuda.memory_reserved())
        if torch.cuda.memory_reserved()>4e9:torch.cuda.empty_cache()
        telemetry=sample_memory(os.getpid(),'TR_only_batch_cache_preparation');reason,observation=resource.observe(telemetry['system_memory'],telemetry['worker_private_memory'],observed_monotonic=telemetry['observed_monotonic'])
        reason=reason or worker_process_commit_reason(telemetry['calling_process_commit_memory'],os.getpid())
        value.update(processed_missing_rows=min(offset+8,len(selected)),last_owned_memory=telemetry,resource_guard=observation,
            peak_allocated_cuda_bytes=max_allocated,peak_observed_reserved_cuda_bytes=max_reserved,elapsed_seconds=time.perf_counter()-started)
        if args.mode=='smoke' or (offset//8)%25==0:write_json(report,value)
        if reason:stop_reason=reason;break
        if args.stop_file and args.stop_file.exists():stop_reason='root_owned_cooperative_stop_file';break
        if args.max_wall_seconds and time.perf_counter()-started>args.max_wall_seconds:stop_reason='bounded_wall_budget';break
    if any(sha(path)!=expected for path,expected in protected.items()):raise ValueError('Protectedexisting/replaycache bytes changed')
    if any(sha(OUTPUT/name)!=expected for name,expected in sources.items()):raise ValueError('Frozen sourcechanged duringcachepreparation')
    passed=stop_reason is None and (len(smoke)==16 if args.mode=='smoke' else len(written)==len(selected))
    value.update(status='complete' if passed else 'stopped_partial_cache_preserved',passed=passed,
        mask_and_label_metadata_parity_passed=passed if args.mode=='smoke' else None,
        existing_bounded_cache_bytes_exact=True,smoke_rows=smoke,new_cache_files=written,stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(report,value);print(json.dumps({'passed':passed,'mode':args.mode,'processed_rows':value.get('processed_missing_rows',0),'elapsed_seconds':value['elapsed_seconds'],'stop_reason':stop_reason}))
    if not passed:raise SystemExit(1)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('plan','smoke','populate'),default='plan')
    p.add_argument('--manifest',type=Path,default=ROOT/'work/data/tr_public_v3_train.jsonl');p.add_argument('--report',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--validated-smoke',type=Path)
    p.add_argument('--root-owned-solo-gpu',action='store_true');p.add_argument('--stop-file',type=Path)
    p.add_argument('--max-wall-seconds',type=float);return p


if __name__=='__main__':main(parser().parse_args())
