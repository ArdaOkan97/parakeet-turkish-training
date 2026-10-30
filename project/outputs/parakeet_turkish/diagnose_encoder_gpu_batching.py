"""Correctness-only singleton/batch encoder trace; no throughput measurement.

This diagnostic may share the GPU with the controller's frozen teacher cache.
It uses identical singleton CPU mel tensors, pads them directly, freezes model
statistics, disables TF32, and traces actual pretrained encoder blocks.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import inspect
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention import SDPBackend, sdpa_kernel

from runtime import ROOT, OUTPUT, load_model, read_audio
from extend_vocab import extend_processor_and_model
from encoder_lora import install_encoder_lora, encoder_lora_metadata

BASE_REVISION="541d1f99c6b0c3cd0b11a95167540bb8edefd82b"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)


def check_memory(limit):
    allocated=torch.cuda.memory_allocated()/1e9
    if allocated>limit:
        raise RuntimeError(f'Diagnostic process allocation {allocated:.3f}GB exceeds {limit:.3f}GB cap')
    return allocated


def specs_and_rows(quartet):
    assessment=json.loads((OUTPUT/'batching_assessment.json').read_text(encoding='utf-8'))
    q=assessment['representative_train_quartets'][quartet]
    specs=[q['tr'][0],q['replay'][0],q['tr'][1],q['replay'][1]]
    lookup={}
    for path in {s['manifest'] for s in specs}:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            row=json.loads(line);lookup[row['source_id']]=row
    rows=[]
    for s in specs:
        row=lookup[s['source_id']]
        if row['dataset_split']!='train' or digest(row['audio_filepath'])!=s['wav_sha256']:
            raise RuntimeError('Diagnostic TRAIN provenance/WAV differs')
        rows.append(row)
    return specs,rows


def tensor_digest(tensor):
    value=tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def pad_singletons(inputs,indices):
    features=pad_sequence([inputs[i]['input_features'][0] for i in indices],batch_first=True,padding_value=0)
    masks=pad_sequence([inputs[i]['attention_mask'][0] for i in indices],batch_first=True,padding_value=0)
    return {'input_features':features,'attention_mask':masks}


def kernel_flags():
    return {'tf32_matmul_allowed':torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn_allowed':torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'cudnn_enabled':torch.backends.cudnn.enabled,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'sdpa_flash_enabled':torch.backends.cuda.flash_sdp_enabled(),
            'sdpa_math_enabled':torch.backends.cuda.math_sdp_enabled(),
            'sdpa_mem_efficient_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),
            'sdpa_cudnn_enabled':torch.backends.cuda.cudnn_sdp_enabled()}


def main(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if not 0<args.max_process_gb<=6:raise ValueError('Diagnostic process cap must lie in (0,6]GB')
    output=OUTPUT/args.output
    if output.exists():raise RuntimeError('Preserve previous diagnostic and choose a fresh output filename')
    free,total=torch.cuda.mem_get_info()
    if free/1e9<4.5:raise RuntimeError('Less than4.5GB free; defer until coordinated GPU window')
    source_names=('diagnose_encoder_gpu_batching.py','runtime.py','extend_vocab.py','encoder_lora.py')
    source_hashes={name:digest(OUTPUT/name) for name in source_names}
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(20261010);torch.cuda.manual_seed_all(20261010)
    model,processor=load_model()
    base_path=Path((ROOT/'work/base_model_path.txt').read_text().strip())
    if base_path.name!=BASE_REVISION:raise RuntimeError('Local base snapshot differs from the pinned revision')
    extend_processor_and_model(processor,model)
    install_encoder_lora(model,rank=8,alpha=16)
    state=torch.load(args.weights,map_location='cpu',weights_only=True)
    expected={name:p for name,p in model.named_parameters() if p.requires_grad}
    if (set(state['weights'])!=set(expected) or state['lora_metadata']!=encoder_lora_metadata(model)
            or state['run_id']!='lora_20261010_01' or state['step']!=1200):
        raise RuntimeError('Diagnostic requires exact learned1200 LoRA layout/state')
    for name,weight in state['weights'].items():
        if weight.shape!=expected[name].shape or weight.dtype!=torch.float32 or not torch.isfinite(weight).all():
            raise RuntimeError('Diagnostic checkpoint shape/dtype/finiteness differs')
    model.load_state_dict(state['weights'],strict=False)
    del state
    model.eval().requires_grad_(False)
    check_memory(args.max_process_gb)
    specs,rows=specs_and_rows(args.quartet)
    inputs=[]
    for row in rows:
        item=dict(processor(audio=[read_audio(row)],sampling_rate=16000,return_tensors='pt'))
        item['input_features']=item['input_features'].masked_fill(~item['attention_mask'].bool().unsqueeze(-1),0)
        inputs.append(item)
    input_metadata=[{'source_id':r['source_id'],'language':r['language'],
                     'input_shape':list(i['input_features'].shape),'valid_mel_frames':int(i['attention_mask'].sum()),
                     'feature_sha256':tensor_digest(i['input_features']),
                     'mask_sha256':tensor_digest(i['attention_mask']),
                     'expected_encoder_frames':s['encoder_frames'],'wav_sha256':s['wav_sha256']}
                    for r,i,s in zip(rows,inputs,specs)]
    report={'schema':1,'status':'running','correctness_only':True,'concurrent_teacher_cache_possible':True,
            'throughput_measured':False,'base_revision':BASE_REVISION,'source_weights_path':str(args.weights),
            'source_weights_sha256':digest(args.weights),'source_hashes':source_hashes,
            'torch_version':torch.__version__,'transformers_version':__import__('transformers').__version__,
            'hardware':torch.cuda.get_device_name(0),'initial_device_free_gb':free/1e9,
            'device_total_gb':total/1e9,'max_process_allocation_gb':args.max_process_gb,
            'model_parameter_dtype_counts':{},'inputs':input_metadata,'cases':[],
            'pinned_base_config_sha256':digest(base_path/'config.json'),
            'installed_modeling_source_sha256':digest(Path(inspect.getfile(model.__class__))),
            'installed_frontend_source_sha256':digest(Path(inspect.getfile(processor.feature_extractor.__class__))),
            'encoder_attention_implementation':model.encoder.config._attn_implementation,
            'scope':'Encoder inference activations only, same learned pilot; no TDT/teacher/backward/Adam/throughput or training-batch approval'}
    for p in model.parameters():
        key=str(p.dtype);report['model_parameter_dtype_counts'][key]=report['model_parameter_dtype_counts'].get(key,0)+p.numel()
    write(output,report)

    def trace(cpu_inputs,expected_lengths):
        gpu_inputs={name:value.cuda() for name,value in cpu_inputs.items()}
        captured={};handles=[]
        def hook(name):
            def capture(module,args,output):
                value=output.detach()
                check_memory(args_outer.max_process_gb)
                captured[name]={'dtype':str(value.dtype),'shape':list(value.shape),
                                'value':value.cpu().float()}
            return capture
        args_outer=args
        handles.append(model.encoder.subsampling.register_forward_hook(hook('subsampling')))
        for index,layer in enumerate(model.encoder.layers):
            handles.append(layer.register_forward_hook(hook(f'encoder_block_{index}')))
        try:
            encoded=model.encoder(**gpu_inputs)
            actual=encoded.attention_mask.sum(-1).cpu().tolist()
            if actual!=expected_lengths:raise RuntimeError('Diagnostic mask lengths differ from original cache')
            captured['encoder_output']={'dtype':str(encoded.last_hidden_state.dtype),
                'shape':list(encoded.last_hidden_state.shape),'value':encoded.last_hidden_state.cpu().float()}
        finally:
            for handle in handles:handle.remove()
        del gpu_inputs,encoded
        return captured

    precisions=('fp32',) if args.fp32_only else ('fp32','bf16')
    modes=('default_sdpa','forced_math','forced_math_cudnn_disabled')
    for precision in precisions:
        for mode in modes:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            before_rng=torch.cuda.get_rng_state().clone()
            contexts=(contextlib.nullcontext() if mode=='default_sdpa' else sdpa_kernel(SDPBackend.MATH))
            cudnn_context=(torch.backends.cudnn.flags(enabled=False,benchmark=False,allow_tf32=False)
                           if mode.endswith('cudnn_disabled') else contextlib.nullcontext())
            autocast=(torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else contextlib.nullcontext())
            with torch.inference_mode(),contexts,cudnn_context,autocast:
                flags=kernel_flags()
                singleton=[trace(value,[specs[index]['encoder_frames']]) for index,value in enumerate(inputs)]
                clips=[]
                for group in ([0,2],[1,3]):
                    batched_input=pad_singletons(inputs,group)
                    batch=trace(batched_input,[specs[index]['encoder_frames'] for index in group])
                    for position,index in enumerate(group):
                        length=specs[index]['encoder_frames'];stages=[]
                        for name in singleton[index]:
                            a=singleton[index][name]['value'][0,:length]
                            b=batch[name]['value'][position,:length]
                            delta=b-a
                            stages.append({'stage':name,'singleton_dtype':singleton[index][name]['dtype'],
                                'batch_dtype':batch[name]['dtype'],'singleton_shape':singleton[index][name]['shape'],
                                'batch_shape':batch[name]['shape'],'relative_l2_difference':float(delta.norm()/a.norm().clamp_min(1e-12)),
                                'max_abs_difference':float(delta.abs().max()),'all_finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())})
                        clips.append({'source_id':rows[index]['source_id'],'language':rows[index]['language'],
                            'valid_encoder_frames':length,'group_indices':group,
                            'valid_input_features_exact_equal':bool(torch.equal(inputs[index]['input_features'][0,:int(inputs[index]['attention_mask'].sum())],
                                batched_input['input_features'][position,:int(inputs[index]['attention_mask'].sum())])),
                            'stages':stages})
                    del batch,batched_input
                del singleton
            rng_equal=bool(torch.equal(before_rng,torch.cuda.get_rng_state()))
            case={'precision':precision,'mode':mode,'kernel_flags':flags,'cuda_rng_unchanged':rng_equal,
                  'peak_process_allocated_gb':torch.cuda.max_memory_allocated()/1e9,
                  'peak_process_reserved_gb':torch.cuda.max_memory_reserved()/1e9,
                  'clips':clips,'maximum_relative_l2_difference':max(s['relative_l2_difference'] for c in clips for s in c['stages'])}
            report['cases'].append(case)
            write(output,report)
            print(json.dumps({'precision':precision,'mode':mode,'max_relative_l2':case['maximum_relative_l2_difference'],
                              'encoder_output_relative_l2':[c['stages'][-1]['relative_l2_difference'] for c in clips],
                              'peak_diagnostic_allocated_gb':case['peak_process_allocated_gb'],'rng_unchanged':rng_equal}),flush=True)
            check_memory(args.max_process_gb)
    report['status']='complete'
    report['all_finite']=all(s['all_finite'] for case in report['cases'] for c in case['clips'] for s in c['stages'])
    report['all_inputs_exact_and_rng_unchanged']=all(case['cuda_rng_unchanged'] and all(c['valid_input_features_exact_equal'] for c in case['clips']) for case in report['cases'])
    if any(digest(OUTPUT/name)!=sha for name,sha in source_hashes.items()):raise RuntimeError('Source changed during diagnostic')
    write(output,report)
    del model,processor
    gc.collect();torch.cuda.empty_cache()
    print(json.dumps({'saved':str(output),'status':'complete','diagnostic_gpu_released':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=ROOT/'work/lora_20261010_01_best.pt')
    parser.add_argument('--quartet',default='median')
    parser.add_argument('--max-process-gb',type=float,default=6.)
    parser.add_argument('--fp32-only',action='store_true')
    parser.add_argument('--output',default='encoder_gpu_batching_diagnostic.json')
    main(parser.parse_args())
