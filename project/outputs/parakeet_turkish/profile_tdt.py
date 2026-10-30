"""Sequential actual-clip GPU profiling for full, reduced, and encoder LoRA TDT.

Run in separate processes after the current training job releases the GPU:
  python profile_tdt.py --mode full
  python profile_tdt.py --mode reduced
  python profile_tdt.py --mode lora
  python profile_tdt.py --mode parity

All modes start from the same FP32 v3 base plus the six appended Turkish rows.
Each profile performs one Adam step accumulating the same long Turkish and replay
recordings. The original cached encoder features feed the frozen teacher. LoRA
uses raw audio, all 24 rank-8 attention adapters, eval-mode encoder statistics,
and explicit nonreentrant block checkpoints. No checkpoint is saved or changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from efficient_tdt import chunked_tdt_loss
from encoder_lora import install_encoder_lora
from extend_vocab import extend_processor_and_model
from metrics import normalize_text
from runtime import ROOT, OUTPUT, inputs_for, load_model
from transformers.loss.loss_tdt import tdt_loss


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def cpu_examples(selection):
    specifications = json.loads(Path(selection).read_text(encoding='utf-8'))
    result = {}
    for name, example in {**specifications['examples'], 'short_parity': specifications['short_parity']}.items():
        row = example['row']
        item = torch.load(example['cache_path'], map_location='cpu', weights_only=True)
        if item['source_id'] != row['source_id']:
            raise ValueError('Cached source ID differs from selected manifest row.')
        features, labels = item['features'], item['labels']
        if features.ndim != 2 or labels.ndim != 1 or not labels.numel() or not features.shape[0]:
            raise ValueError('Selected cache must have positive T and U and correct tensor ranks.')
        if not torch.isfinite(features).all() or (labels < 0).any():
            raise ValueError('Selected cache contains invalid features or targets.')
        wav = Path(row['audio_filepath'])
        if row.get('wav_sha256') and hashlib.sha256(wav.read_bytes()).hexdigest() != row['wav_sha256']:
            raise ValueError('Selected prepared WAV hash differs from its manifest.')
        result[name] = {'row': row, 'item': item, 'cache_path': example['cache_path'],
            'T': int(features.shape[0]), 'U': int(labels.numel()),
            'lattice_states': int(features.shape[0] * (labels.numel()+1))}
    if result['tr']['row']['language'] != 'tr' or result['replay']['row']['language'] == 'tr':
        raise ValueError('Profile needs one Turkish and one retention-language recording.')
    return result


def checkpoint_encoder_blocks(model):
    """HF's native checkpoint path is gated on train(); keep encoder eval()."""
    for layer in model.encoder.layers:
        original = layer.forward
        def wrapped(hidden_states, *args, _forward=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_forward, hidden_states, *args, use_reentrant=False, **kwargs)
            return _forward(hidden_states, *args, **kwargs)
        layer.gradient_checkpointing = False
        layer.forward = wrapped


def initialize(args, examples):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, processor = load_model(args.base_path)
    old_size = int(model.config.vocab_size)
    teacher = {name: copy.deepcopy(getattr(model, name)).eval().requires_grad_(False)
               for name in ('encoder_projector', 'decoder', 'joint')}
    extension = extend_processor_and_model(processor, model)
    for value in examples.values():
        row, item = value['row'], value['item']
        labels = processor.tokenizer(row['text'], return_tensors='pt')['input_ids'][0]
        if not torch.equal(labels, item['labels'].long()):
            raise ValueError(f"Current extended tokenizer differs from cached labels: {row['source_id']}")
        decoded = processor.tokenizer.decode(labels, skip_special_tokens=True)
        if normalize_text(decoded, row['language']) != normalize_text(row['text'], row['language']):
            raise ValueError('Selected text is not representable under the shared lexical normalizer.')
        if row['language'] != 'tr' and int(labels.max()) >= old_size:
            raise ValueError('Replay teacher cannot embed a newly appended Turkish token ID.')
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    lora = None
    if args.mode == 'lora':
        lora = install_encoder_lora(model, rank=8, alpha=16.0)
        if len(lora['adapted_layer_indices']) != 24:
            raise ValueError('Expected all 24 encoder blocks to have attention adapters.')
        checkpoint_encoder_blocks(model)
    model.train()
    model.encoder.eval()
    return model, processor, teacher, old_size, extension, lora


def prepare_states(model, processor, example, mode):
    item, row = example['item'], example['row']
    cached_features = item['features'][None].cuda().float()
    labels = item['labels'][None].cuda().long()
    decoder_ids = torch.cat([torch.full((1,1), model.config.blank_token_id, device='cuda', dtype=torch.long), labels], dim=1)
    if mode == 'lora':
        inputs = inputs_for(processor, row)
        encoded = model.encoder(**inputs)
        length = int(encoded.attention_mask.sum())
        features = encoded.last_hidden_state[:, :length]
        if length != cached_features.shape[1]:
            raise ValueError('Student raw-audio encoder length differs from the original teacher cache.')
    else:
        features = cached_features
    ep = model.encoder_projector(features)
    dp = model.decoder(decoder_ids)
    return ep, dp, labels, decoder_ids, cached_features


def sampled_kl(model, teacher, cached_features, decoder_ids, ep, dp, old_size, full_logits=None):
    ts = torch.linspace(0, ep.shape[1]-1, min(8,ep.shape[1]), device='cuda').long()
    us = torch.linspace(0, dp.shape[1]-1, min(8,dp.shape[1]), device='cuda').long()
    with torch.no_grad():
        original_ep = teacher['encoder_projector'](cached_features)[:,ts]
        original_dp = teacher['decoder'](decoder_ids)[:,us]
        target = teacher['joint'](original_ep[:,:,None], original_dp[:,None])[...,:old_size].float().softmax(-1)
    if full_logits is None:
        student_logits = model.joint(ep[:,ts,None],dp[:,None,us])[...,:model.config.vocab_size]
    else:
        student_logits = full_logits[:,ts][:,:,us,:model.config.vocab_size]
    student = student_logits.float().log_softmax(-1)
    return (target*(target.clamp_min(1e-12).log()-student[...,:old_size])).sum(-1).mean()


def objective(model, processor, teacher, old_size, example, mode, args):
    ep, dp, labels, decoder_ids, cached_features = prepare_states(model, processor, example, mode)
    lengths = torch.tensor([ep.shape[1]], device='cuda', dtype=torch.long)
    target_lengths = torch.tensor([labels.shape[1]], device='cuda', dtype=torch.long)
    logits = None
    if mode == 'full':
        logits = model.joint(ep[:,:,None],dp[:,None])
        tdt = tdt_loss(logits[...,:model.config.vocab_size], logits[...,model.config.vocab_size:],
            labels, lengths, target_lengths, model.config.blank_token_id, model.config.durations,
            sigma=args.sigma, reduction='mean')
    else:
        tdt = chunked_tdt_loss(model.joint, ep, dp, labels, model.config.vocab_size,
            model.config.blank_token_id, model.config.durations, lengths, target_lengths,
            chunk_size=args.chunk_size, sigma=args.sigma, reduction='mean', recompute=True)
    kl = torch.zeros((),device='cuda') if example['row']['language']=='tr' else sampled_kl(
        model,teacher,cached_features,decoder_ids,ep,dp,old_size,logits)
    return tdt+args.replay_kl*kl, tdt, kl


def memory():
    return {'allocated_bytes':torch.cuda.memory_allocated(), 'reserved_bytes':torch.cuda.memory_reserved(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def gradient_audit(model):
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    missing=[name for name,p in trainable if p.grad is None]
    nonfinite=[name for name,p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
    frozen_with_grad=[name for name,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    adapter_nonzero={suffix:sum(bool(p.grad is not None and p.grad.abs().sum()>0) for name,p in trainable if name.endswith(suffix))
                     for suffix in ('lora_A','lora_B')}
    return {'trainable_parameters':sum(p.numel() for _,p in trainable), 'trainable_tensors':len(trainable),
        'missing_gradient_names':missing, 'nonfinite_gradient_names':nonfinite,
        'frozen_parameter_gradient_names':frozen_with_grad, 'adapter_nonzero_gradient_tensors':adapter_nonzero,
        'all_trainable_gradients_finite_and_present':not missing and not nonfinite and not frozen_with_grad}


def one_step(model,processor,teacher,old_size,examples,args):
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start_allocated=memory(); start=time.perf_counter(); details=[]
    for name in ('tr','replay'):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            total,tdt,kl=objective(model,processor,teacher,old_size,examples[name],args.mode,args)
            scaled=total/2
        if not torch.isfinite(scaled):
            raise RuntimeError(f'Nonfinite profile loss for {name}')
        details.append({'stream':name,'tdt_loss':float(tdt.detach()),'replay_kl':float(kl.detach()),
                        'weighted_total_before_accumulation_division':float(total.detach())})
        scaled.backward()
        del total,tdt,kl,scaled
    gradients=gradient_audit(model)
    if not gradients['all_trainable_gradients_finite_and_present']:
        raise RuntimeError(f'Invalid gradient audit: {gradients}')
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
    optimizer.step(); torch.cuda.synchronize()
    return {'status':'complete','seconds_one_accumulated_adam_step':time.perf_counter()-start,
        'microbatches':2,'optimizer_steps':1,'optimizer':'AdamW FP32 parameters/states, learning_rate=1e-5, weight_decay=.01',
        'gradient_norm_before_clip':float(norm),'gradient_audit':gradients,'objectives':details,
        'memory_at_step_start':start_allocated,'memory_after_step':memory(),
        'peak_includes':'Base FP32 model, frozen teacher heads, gradients, two sequential backward passes and newly allocated Adam states.'}


def parity(model,processor,teacher,old_size,example,args):
    summaries={}; reference={}; parameter_details=[]
    for mode in ('full','reduced'):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,tdt,kl=objective(model,processor,teacher,old_size,example,mode,args)
        loss.backward();torch.cuda.synchronize()
        audit=gradient_audit(model)
        summaries[mode]={'weighted_loss':float(loss.detach()),'tdt_loss':float(tdt.detach()),'kl':float(kl.detach()),
            'seconds_forward_backward':time.perf_counter()-start,'gradient_audit':audit,'memory':memory()}
        del loss,tdt,kl
        if not audit['all_trainable_gradients_finite_and_present']:
            raise RuntimeError(f'Invalid parity gradients: {audit}')
        for name,parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            current=parameter.grad.detach().cpu().float()
            if mode=='full':
                reference[name]=current.clone()
            else:
                original=reference[name]
                error=(current-original).double()
                a=original.double(); b=current.double()
                norm_a=float(torch.linalg.vector_norm(a)); norm_b=float(torch.linalg.vector_norm(b))
                difference=float(torch.linalg.vector_norm(error))
                cosine=float((a.flatten()*b.flatten()).sum()/(norm_a*norm_b)) if norm_a and norm_b else (1.0 if difference==0 else 0.0)
                parameter_details.append({'name':name,'maximum_absolute_difference':float(error.abs().max()),
                    'reference_gradient_l2':norm_a,'difference_l2':difference,
                    'relative_gradient_l2':difference/max(norm_a,1e-12),'cosine_similarity':cosine,
                    'torch_allclose_rtol_0_02_atol_0_0001':bool(torch.allclose(current,original,rtol=.02,atol=1e-4))})
    norm=(sum(p['reference_gradient_l2']**2 for p in parameter_details))**0.5
    delta=(sum(p['difference_l2']**2 for p in parameter_details))**0.5
    # Calculate global cosine from norm identities to avoid another large concatenation.
    fullnorm2=sum(p['reference_gradient_l2']**2 for p in parameter_details)
    reducednorm2=sum(float(model.get_parameter(p['name']).grad.detach().double().square().sum()) for p in parameter_details)
    cosine=(fullnorm2+reducednorm2-delta**2)/(2*(fullnorm2*reducednorm2)**0.5)
    loss_difference=abs(summaries['full']['weighted_loss']-summaries['reduced']['weighted_loss'])
    tdt_difference=abs(summaries['full']['tdt_loss']-summaries['reduced']['tdt_loss'])
    significant=[p for p in parameter_details if p['reference_gradient_l2']>1e-8]
    passed=(loss_difference<1e-4 and tdt_difference<1e-4 and delta/max(norm,1e-12)<.02
            and cosine>.999 and all(p['relative_gradient_l2']<.02 and p['cosine_similarity']>.999 for p in significant))
    return {'status':'passed' if passed else 'failed','objective_summaries':summaries,
        'weighted_loss_absolute_difference':loss_difference,'tdt_loss_absolute_difference':tdt_difference,
        'all_head_gradient_relative_l2_difference':delta/max(norm,1e-12),'all_head_gradient_cosine':cosine,
        'per_parameter':parameter_details,'criteria':'Loss and TDT differences <1e-4; global and each non-negligible head gradient relative L2 <.02, cosine >.999.',
        'scope':'Same actual cached clip, unchanged model weights, BF16 autocast, no optimizer step or saved checkpoint.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',choices=('full','reduced','lora','parity'),required=True)
    parser.add_argument('--selection',type=Path,default=OUTPUT/'memory_profile_examples.json')
    parser.add_argument('--base-path')
    parser.add_argument('--chunk-size',type=int,default=16)
    parser.add_argument('--sigma',type=float,default=0.0)
    parser.add_argument('--replay-kl',type=float,default=5.0)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    examples=cpu_examples(args.selection)
    result={'mode':args.mode,'status':'prepared_cpu_only','seed':args.seed,'chunk_size':args.chunk_size,
        'sigma':args.sigma,'replay_kl_coefficient':args.replay_kl,
        'selection_sha256':hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        'code_sha256':{name:hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() for name in
                       ('profile_tdt.py','efficient_tdt.py','encoder_lora.py','extend_vocab.py','runtime.py')},
        'examples':{name:{'source_id':example['row']['source_id'],'language':example['row']['language'],
            'cache_path':example['cache_path'],'T':example['T'],'U':example['U'],
            'lattice_states':example['lattice_states']} for name,example in examples.items()},
        'dtype':'FP32 base, adapter and optimizer parameters; CUDA BF16 autocast; FP32 token/duration log probabilities',
        'teacher_policy':'Frozen original teacher heads; original cached encoder features; replay targets have no appended IDs.',
        'timing_policy':'Synchronized CUDA wall time; model and teacher loading excluded; optimizer-state allocation included in profile step.',
        'comparison_policy':'Run each mode in a separate process on an otherwise idle GPU. Torch allocator peaks exclude driver/other-process allocations.'}
    output=args.output or OUTPUT/f'tdt_profile_{args.mode}.json'
    if args.prepare_only:
        write_json(output.with_name(output.stem+'_plan.json'),result)
        print(json.dumps(result,indent=2),flush=True)
        return
    model,processor,teacher,old_size,extension,lora=initialize(args,examples)
    result.update(hardware=torch.cuda.get_device_name(0),vocabulary_extension=extension,encoder_lora=lora,
        base_path=args.base_path or (ROOT/'work/base_model_path.txt').read_text().strip())
    try:
        result['measurement']=parity(model,processor,teacher,old_size,examples['short_parity'],args) if args.mode=='parity' else one_step(
            model,processor,teacher,old_size,examples,args)
        result['status']=result['measurement']['status']
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',memory_at_failure=memory())
        write_json(output,result)
        raise
    write_json(output,result)
    print(json.dumps({key:result[key] for key in ('mode','status','measurement')},indent=2),flush=True)
    if result['status']=='failed':
        raise RuntimeError('GPU parity did not satisfy its saved acceptance criteria.')


if __name__=='__main__':
    main()
