"""Encoder-attention LoRA plus Turkish heads, with memory-efficient TDT loss.

Training uses raw audio for adapted encoder gradients. Cached original encoder
features condition the fixed teacher only. All checkpoint selection is dev-only;
held-out benchmarking is a separate finalization step.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from runtime import ROOT, DATA, OUTPUT, load_model, inputs_for
from extend_vocab import extend_processor_and_model
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from efficient_tdt import chunked_tdt_loss
from train import make_cache, checkpoint_state
from benchmark import read_jsonl, write_json, evaluate, LANGUAGES
from metrics import normalize_text


def checkpoint_encoder_blocks(model):
    """Recompute activations without enabling dropout or BatchNorm updates."""
    for layer in model.encoder.layers:
        original = layer.forward
        def wrapped(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)
        layer.forward = wrapped


class CyclingRows:
    def __init__(self, rows, rng):
        self.rows = list(rows)
        self.rng = rng
        self.position = len(rows)
        self.draws = 0
        self.seen = set()
    def next(self):
        if self.position >= len(self.rows):
            self.rng.shuffle(self.rows)
            self.position = 0
        row = self.rows[self.position]
        self.position += 1
        self.draws += 1
        self.seen.add(row['source_id'])
        return row


def preservation_losses(model, teacher, base_features, encoder_projected,
                         decoder_projected, decoder_ids, old_size):
    ts = torch.linspace(0, encoder_projected.shape[1]-1, min(8, encoder_projected.shape[1]), device='cuda').long()
    us = torch.linspace(0, decoder_projected.shape[1]-1, min(8, decoder_projected.shape[1]), device='cuda').long()
    with torch.no_grad():
        original_ep = teacher['encoder_projector'](base_features)[:, ts]
        original_dp = teacher['decoder'](decoder_ids)[:, us]
        original_logits = teacher['joint'](original_ep[:, :, None], original_dp[:, None])
        tokens = original_logits[..., :old_size].float().softmax(-1)
        durations = original_logits[..., old_size:].float().softmax(-1)
    logits = model.joint(encoder_projected[:, ts, None], decoder_projected[:, None, us])
    log_tokens = logits[..., :model.config.vocab_size].float().log_softmax(-1)
    log_durations = logits[..., model.config.vocab_size:].float().log_softmax(-1)
    token_kl = (tokens * (tokens.clamp_min(1e-12).log() - log_tokens[..., :old_size])).sum(-1).mean()
    duration_kl = (durations * (durations.clamp_min(1e-12).log() - log_durations)).sum(-1).mean()
    return token_kl, duration_kl


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, processor = load_model()
    baseline = json.loads((OUTPUT/'results/base_v3/summary.json').read_text(encoding='utf-8'))
    for lang in LANGUAGES:
        wanted = {r['source_id'] for r in read_jsonl(DATA/f'validation_{lang}.jsonl')}
        found = {r['source_id'] for r in read_jsonl(OUTPUT/f'results/base_v3/validation_{lang}.jsonl')}
        if wanted != found or baseline[f'validation_{lang}']['asr']['utterances'] != len(wanted):
            raise RuntimeError('Selection requires a complete fixed development baseline')
    tr = read_jsonl(DATA/'tr_train.jsonl')
    random.shuffle(tr)
    tr = tr[:args.train_limit]
    if args.supplement:
        supplement = read_jsonl(args.supplement)
        tr += supplement
    if len({r['source_id'] for r in tr}) != len(tr):
        raise RuntimeError('Duplicate Turkish training IDs')
    replay = read_jsonl(DATA/'replay_train.jsonl')
    heldout = [r for lang in LANGUAGES for split in ('validation', 'test') for r in read_jsonl(DATA/f'{split}_{lang}.jsonl')]
    heldout_ids = {r['source_id'] for r in heldout}
    heldout_sentences = {r['sentence_id'] for r in heldout if r.get('sentence_id') is not None}
    heldout_audio = {r['audio_sha256'] for r in heldout}
    if any(r['source_id'] in heldout_ids or r['audio_sha256'] in heldout_audio
           or (r.get('sentence_id') is not None and r['sentence_id'] in heldout_sentences) for r in tr+replay):
        raise RuntimeError('Training/held-out provenance overlaps')
    teacher = {name:copy.deepcopy(getattr(model,name)).eval() for name in ('encoder_projector','decoder','joint')}
    for module in teacher.values():
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    old_size = model.config.vocab_size
    audit = extend_processor_and_model(processor, model, [r['text'] for r in tr])
    if args.phonetic_sh_donors:
        # Romanian comma-below ș has the /ʃ/ sound of Turkish cedilla ş.
        for new, donor in [('ş','ș'),('Ş','Ș')]:
            new_id = processor.tokenizer.convert_tokens_to_ids(new)
            donor_id = processor.tokenizer.convert_tokens_to_ids(donor)
            if new_id < old_size or not 0 <= donor_id < old_size:
                raise RuntimeError('Phonetic donor lookup failed')
            with torch.no_grad():
                model.decoder.embedding.weight[new_id].copy_(model.decoder.embedding.weight[donor_id])
                model.joint.head.weight[new_id].copy_(model.joint.head.weight[donor_id])
                model.joint.head.bias[new_id].copy_(model.joint.head.bias[donor_id]-2)
            audit.setdefault('phonetic_donor_override',[]).append({'new':new,'new_id':new_id,'donor':donor,'donor_id':donor_id})
    write_json(OUTPUT/'vocabulary_audit.json',audit)
    def lexical_roundtrip(row):
        ids = processor.tokenizer(row['text'])['input_ids']
        text = processor.tokenizer.decode(ids,skip_special_tokens=True)
        return (ids and len(ids)<=args.max_tokens
                and normalize_text(text,row['language'])==normalize_text(row['text'],row['language']))
    initial_counts = {'tr':len(tr),'replay':len(replay)}
    tr = [r for r in tr if lexical_roundtrip(r)]
    replay = [r for r in replay if lexical_roundtrip(r)
              and max(processor.tokenizer(r['text'])['input_ids'])<old_size]
    namespace = hashlib.sha256(('541d1f99c6b0c3cd0b11a95167540bb8edefd82b'+processor.tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
    cache = ROOT/'work/encoder-cache'/namespace
    # Freeze all original encoder parameters while building the teacher cache.
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    tr = make_cache(model,processor,tr,cache,args.max_tokens)
    replay = make_cache(model,processor,replay,cache,args.max_tokens)
    by_lang = {lang:[r for r in replay if r['language']==lang] for lang in LANGUAGES if lang!='tr'}
    if not tr or any(not rows for rows in by_lang.values()):
        raise RuntimeError('Empty training stream')
    layout = install_encoder_lora(model,rank=args.rank,alpha=args.alpha,last_n_layers=args.last_n_layers)
    model.config.encoder_lora_tracking = {
        'layout':layout,'implementation_sha256':hashlib.sha256((OUTPUT/'encoder_lora.py').read_bytes()).hexdigest(),
        'kind':'unmerged'}
    checkpoint_encoder_blocks(model)
    if args.resume_weights:
        saved = torch.load(args.resume_weights,map_location='cpu',weights_only=True)
        expected = {name for name,parameter in model.named_parameters() if parameter.requires_grad}
        if set(saved['weights']) != expected:
            raise RuntimeError('Resume checkpoint must contain exactly this model layout\'s trainable weights')
        model.load_state_dict(saved['weights'],strict=False)
    adapters = [p for n,p in model.named_parameters() if p.requires_grad and n.endswith(('lora_A','lora_B'))]
    heads = [p for n,p in model.named_parameters() if p.requires_grad and not n.endswith(('lora_A','lora_B'))]
    all_trainable = adapters+heads
    optimizer = torch.optim.AdamW([{'params':heads,'lr':args.head_lr,'weight_decay':.01},
                                  {'params':adapters,'lr':args.lora_lr,'weight_decay':0.0}])
    new_params = (model.decoder.embedding.weight,model.joint.head.weight,model.joint.head.bias)
    rng = random.Random(args.seed+1)
    tr_stream = CyclingRows(tr,rng)
    replay_streams = {lang:CyclingRows(rows,rng) for lang,rows in by_lang.items()}
    run_dir = OUTPUT/'runs'/args.run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    arguments = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config = {**arguments,'base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b',
              'transformers_commit':'536ecc007387a50e77603bb5d92100e9b07514cc',
              'method':'encoder q/k/v/o LoRA plus direct decoder/joint adaptation',
              'encoder_base_frozen':True,'encoder_frozen':False,'batchnorm_stats_frozen':True,
              'encoder_dropout_and_layerdrop_disabled':True,'encoder_activation_checkpointing':True,
              'joint_projection_time_chunk_size':args.chunk_size,'full_vocabulary_tdt_objective':True,
              'trainable_parameters':sum(p.numel() for p in all_trainable),
              'encoder_adapter_parameters':sum(p.numel() for p in adapters),'lora_layout':layout,
              'turkish_clips_used':len(tr),'replay_clips_used':len(replay),
              'excluded_training_counts':{key:initial_counts[key]-len(rows) for key,rows in [('tr',tr),('replay',replay)]},
              'replay_fraction':.5,'sampling':'without-replacement shuffled cycles; retention language uniformly selected',
              'teacher':'original frozen heads conditioned on cached original frozen encoder features',
              'hardware':torch.cuda.get_device_name(0),'weight_dtype':'float32','compute_dtype':'bfloat16 autocast',
              'selection_split':'FLEURS validation only','heldout_test_policy':'separate finalization after dev selection',
              'retention_gate':'each dev WER <= base + max(0.5 percentage point,10% relative base WER)',
              'code_sha256':{name:hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() for name in
                             ('train_lora.py','encoder_lora.py','efficient_tdt.py','extend_vocab.py')}}
    write_json(run_dir/'run_config.json',config)
    write_json(OUTPUT/'run_config.json',config)
    write_json(run_dir/'vocabulary_audit.json',audit)
    history = []
    best = math.inf
    selected_step = None
    best_path = ROOT/'work'/f'{args.run_id}_best.pt'
    latest_path = ROOT/'work'/f'{args.run_id}_latest.pt'
    model.train()
    model.encoder.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print(f'Encoder LoRA params {sum(p.numel() for p in adapters):,}; heads {sum(p.numel() for p in heads):,}; Turkish {len(tr)}; replay {len(replay)}',flush=True)
    for step in range(1,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        tdt_sum = kl_sum = duration_sum = feature_sum = 0.0
        # Warm up both parameter groups over 25 updates.
        warm = min(1.0,step/25)
        optimizer.param_groups[0]['lr'] = args.head_lr*warm
        optimizer.param_groups[1]['lr'] = args.lora_lr*warm
        for micro in range(args.accumulation):
            row = tr_stream.next() if micro%2==0 else replay_streams[rng.choice(list(replay_streams))].next()
            cached = torch.load(row['cache_path'],map_location='cpu',weights_only=True)
            labels = cached['labels'][None].cuda().long()
            ids = torch.cat([torch.full((1,1),model.config.blank_token_id,device='cuda',dtype=torch.long),labels],dim=1)
            inputs = inputs_for(processor,row)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                encoded = model.encoder(**inputs)
                frame_length = int(encoded.attention_mask.sum())
                features = encoded.last_hidden_state[:, :frame_length]
                ep = model.encoder_projector(features)
                dp = model.decoder(ids)
                tdt = chunked_tdt_loss(model.joint,ep,dp,labels,model.config.vocab_size,
                                       model.config.blank_token_id,model.config.durations,
                                       logit_lengths=encoded.attention_mask.sum(-1).long(),chunk_size=args.chunk_size)
                kl = duration_kl = feature_loss = torch.zeros((),device='cuda')
                if row['language']!='tr':
                    base = cached['features'][None].cuda().float()
                    if base.shape != features.shape:
                        raise RuntimeError('Original and adapted encoder frames must align')
                    kl,duration_kl = preservation_losses(model,teacher,base,ep,dp,ids,old_size)
                    feature_loss = (1-F.cosine_similarity(features.float(),base,dim=-1)).mean()
                loss = (tdt+args.replay_kl*kl+args.duration_kl*duration_kl+args.feature_distillation*feature_loss)/args.accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at {step} {row["source_id"]}')
            loss.backward()
            tdt_sum += float(tdt.detach())
            kl_sum += float(kl.detach())
            duration_sum += float(duration_kl.detach())
            feature_sum += float(feature_loss.detach())
            del encoded,features,ep,dp,tdt,kl,duration_kl,feature_loss,loss,inputs
        norm = torch.nn.utils.clip_grad_norm_(all_trainable,1.0,error_if_nonfinite=True)
        previous = [p[old_size:model.config.vocab_size].detach().clone() for p in new_params]
        optimizer.step()
        with torch.no_grad():
            for p,old in zip(new_params,previous):
                rows = p[old_size:model.config.vocab_size]
                rows.copy_(old+args.new_token_lr_multiplier*(rows-old))
        record = {'step':step,'tdt_loss':tdt_sum/args.accumulation,'replay_kl':kl_sum/args.accumulation,
                  'replay_duration_kl':duration_sum/args.accumulation,'replay_feature_distillation':feature_sum/args.accumulation,
                  'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-start,
                  'peak_allocated_vram_gb':torch.cuda.max_memory_allocated()/1e9,
                  'current_allocated_vram_gb':torch.cuda.memory_allocated()/1e9,
                  'current_reserved_vram_gb':torch.cuda.memory_reserved()/1e9,
                  'turkish_unique_clips_seen':len(tr_stream.seen)}
        history.append(record)
        if step%10==0:
            print(json.dumps(record),flush=True)
            write_json(OUTPUT/'training_history.json',history)
            write_json(run_dir/'training_history.json',history)
        if step%args.eval_every==0 or step==args.steps or step==args.first_eval:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            dev = evaluate(model,processor,f'{args.run_id}_step_{step}',splits=('validation',))
            deltas = {lang:dev[f'validation_{lang}']['asr']['wer']-baseline[f'validation_{lang}']['asr']['wer'] for lang in by_lang}
            passed = all(deltas[lang]<=max(.005,.1*baseline[f'validation_{lang}']['asr']['wer']) for lang in by_lang)
            score = dev['validation_tr']['asr']['wer']
            record['validation'] = {'turkish_wer':score,'retention_wer_deltas':deltas,'retention_gate_passed':passed}
            state = {'step':step,'weights':checkpoint_state(model),'run_id':args.run_id,'lora_metadata':encoder_lora_metadata(model)}
            torch.save(state,latest_path)
            if passed and score < best and score < baseline['validation_tr']['asr']['wer']:
                best = score
                selected_step = step
                torch.save(state,best_path)
            print(f'dev step{step} Turkish WER {score:.4f}; retention passed {passed}; best {selected_step}',flush=True)
            write_json(OUTPUT/'training_history.json',history)
            write_json(run_dir/'training_history.json',history)
            model.train()
            model.encoder.eval()
            torch.cuda.empty_cache()
    selection = {'accepted':selected_step is not None,'run_id':args.run_id,'selected_step':selected_step,
                 'turkish_validation_wer':best if selected_step is not None else None,
                 'selection_split':'FLEURS validation only','pilot':True,'heldout_tests_evaluated':False,
                 'weights':str(best_path if selected_step is not None else latest_path),
                 'reason':'Selected best Turkish dev WER within every retention gate.' if selected_step is not None
                          else 'No checkpoint improved Turkish and passed all development retention gates.'}
    write_json(run_dir/'selection.json',selection)
    write_json(OUTPUT/'selection.json',selection)
    print(json.dumps(selection),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--eval-every',type=int,default=200)
    parser.add_argument('--first-eval',type=int,default=100)
    parser.add_argument('--train-limit',type=int,default=2000)
    parser.add_argument('--max-tokens',type=int,default=180)
    parser.add_argument('--supplement',type=Path)
    parser.add_argument('--head-lr',type=float,default=2e-5)
    parser.add_argument('--lora-lr',type=float,default=3e-4)
    parser.add_argument('--new-token-lr-multiplier',type=float,default=15.0)
    parser.add_argument('--replay-kl',type=float,default=5.0)
    parser.add_argument('--duration-kl',type=float,default=.5)
    parser.add_argument('--feature-distillation',type=float,default=1.0)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--alpha',type=float,default=16)
    parser.add_argument('--last-n-layers',type=int)
    parser.add_argument('--chunk-size',type=int,default=16)
    parser.add_argument('--phonetic-sh-donors',action='store_true')
    parser.add_argument('--resume-weights',type=Path)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--run-id',default='lora_20261010_01')
    args=parser.parse_args()
    if args.accumulation<2 or args.accumulation%2:
        raise ValueError('50:50 replay requires even accumulation >=2')
    run(args)
