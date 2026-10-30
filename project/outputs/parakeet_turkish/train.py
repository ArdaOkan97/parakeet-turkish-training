"""Conservative frozen-encoder TDT adaptation with 50:50 multilingual replay.

Native Windows: cache frozen acoustic features, train FP32 decoder/joint weights
under bf16 autocast, preserve all encoder statistics, and select on dev only.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
import uuid
from pathlib import Path
import torch
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderModelOutput
from runtime import ROOT, DATA, OUTPUT, load_model, inputs_for
from extend_vocab import extend_processor_and_model
from benchmark import read_jsonl, write_json, evaluate, LANGUAGES
from metrics import paired_bootstrap_delta, normalize_text

def train_mode(model):
    model.train()
    model.encoder.eval()  # freezes BatchNorm stats and disables layerdrop/dropout

def make_cache(model,processor,rows,cache_dir,max_tokens):
    cache_dir.mkdir(parents=True,exist_ok=True)
    model.eval(); accepted=[]
    for index,row in enumerate(rows):
        label=processor.tokenizer(row['text'],return_tensors='pt')['input_ids']
        if label.shape[-1]>max_tokens or label.shape[-1]==0: continue
        key=hashlib.sha256(row['source_id'].encode()).hexdigest()
        path=cache_dir/f'{key}.pt'
        if not path.exists():
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                inputs=inputs_for(processor,row)
                features=model.encoder(**inputs)
                length=int(features.attention_mask.sum())
                item={'features':features.last_hidden_state[0,:length].cpu().contiguous(),
                      'labels':label[0].cpu(),'source_id':row['source_id']}
            torch.save(item,path)
        else:
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if not torch.equal(saved['labels'],label[0].cpu()):
                raise RuntimeError('Cached targets do not match the current tokenizer')
        accepted.append({**row,'cache_path':str(path)})
        if (index+1)%100==0:
            print(f'acoustic cache {index+1}/{len(rows)}',flush=True)
    return accepted

def forward_cached(model,item):
    features=item['features'][None].cuda().float()
    labels=item['labels'][None].cuda().long()
    decoder_ids=torch.cat([torch.full((1,1),model.config.blank_token_id,device='cuda',dtype=torch.long),labels],dim=1)
    mask=torch.ones(features.shape[:2],device='cuda',dtype=torch.bool)
    projected=model.encoder_projector(features)
    encoded=ParakeetEncoderModelOutput(last_hidden_state=features,pooler_output=projected,attention_mask=mask)
    out=model(encoder_outputs=encoded,labels=labels,decoder_input_ids=decoder_ids)
    return out,features,decoder_ids

def teacher_kl(model,teacher,features,decoder_ids,student_logits,old_size):
    # Sparse teacher states cover the lattice without materializing another full
    # vocabulary lattice. KL includes the new-token normalization in the student.
    ts=torch.linspace(0,features.shape[1]-1,min(8,features.shape[1]),device='cuda').long()
    us=torch.linspace(0,decoder_ids.shape[1]-1,min(8,decoder_ids.shape[1]),device='cuda').long()
    with torch.no_grad():
        ep=teacher['encoder_projector'](features)[:,ts]
        dp=teacher['decoder'](decoder_ids)[:,us]
        logits=teacher['joint'](ep[:,:,None,:],dp[:,None,:,:])
        target=logits[...,:old_size].float().softmax(-1)
    student=student_logits[:,ts][:,:,us,:model.config.vocab_size].float().log_softmax(-1)
    return (target*(target.clamp_min(1e-12).log()-student[...,:old_size])).sum(-1).mean()

def checkpoint_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def run(args):
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model,processor=load_model()
    baseline_path=OUTPUT/'results/base_v3/summary.json'
    if not baseline_path.exists():
        raise RuntimeError('Run benchmark.py --tag base_v3 before training.')
    baseline=json.loads(baseline_path.read_text())
    for lang in LANGUAGES:
        if f'validation_{lang}' not in baseline:
            raise RuntimeError('Need separate baseline development scores for all languages.')
        expected={r['source_id'] for r in read_jsonl(DATA/f'validation_{lang}.jsonl')}
        observed={r['source_id'] for r in read_jsonl(OUTPUT/f'results/base_v3/validation_{lang}.jsonl')}
        if expected!=observed or baseline[f'validation_{lang}']['asr']['utterances']!=len(expected):
            raise RuntimeError('Baseline must cover the entire fixed development manifest')
    tr=read_jsonl(DATA/'tr_train.jsonl'); replay=read_jsonl(DATA/'replay_train.jsonl')
    random.shuffle(tr);tr=tr[:args.train_limit]
    # Teacher heads use original tokenizer/labels for replay (IDs are preserved).
    old_size=model.config.vocab_size
    teacher={name:copy.deepcopy(getattr(model,name)).eval() for name in ('encoder_projector','decoder','joint')}
    for module in teacher.values():
        for p in module.parameters():p.requires_grad_(False)
    audit=extend_processor_and_model(processor,model,[r['text'] for r in tr])
    write_json(OUTPUT/'vocabulary_audit.json',audit)
    # Rare foreign proper names in replay can contain newly supported Turkish
    # characters. The original teacher cannot embed these appended IDs.
    before_replay_count=len(replay)
    replay=[r for r in replay if max(processor.tokenizer(r['text'])['input_ids'],default=0)<old_size]
    replay_new_token_exclusions=before_replay_count-len(replay)
    # The official HF tokenizer has digit/whitespace round-trip edge cases.
    # Exclude unrepresentable lexical training targets instead of teaching a
    # transcript different from the human reference. Punctuation/case alone
    # need not round-trip because the paired ASR score deliberately ignores them.
    def lexical_roundtrip(row):
        ids=processor.tokenizer(row['text'])['input_ids']
        decoded=processor.tokenizer.decode(ids,skip_special_tokens=True)
        return normalize_text(decoded,row['language'])==normalize_text(row['text'],row['language'])
    original_tr_count,original_replay_count=len(tr),len(replay)
    tr=[r for r in tr if lexical_roundtrip(r)]
    replay=[r for r in replay if lexical_roundtrip(r)]
    roundtrip_exclusions={'tr':original_tr_count-len(tr),'replay':original_replay_count-len(replay)}
    for p in model.parameters():p.requires_grad_(True)
    for p in model.encoder.parameters():p.requires_grad_(False)
    namespace=hashlib.sha256(('541d1f99c6b0c3cd0b11a95167540bb8edefd82b'+
        processor.tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
    cache_dir=ROOT/'work/encoder-cache'/namespace
    tr=make_cache(model,processor,tr,cache_dir,args.max_tokens)
    replay=make_cache(model,processor,replay,cache_dir,args.max_tokens)
    by_lang={lang:[r for r in replay if r['language']==lang] for lang in LANGUAGES if lang!='tr'}
    if not tr or any(not rows for rows in by_lang.values()):raise RuntimeError('Empty training stream')
    # Split new embedding/output rows into their own trainable tensors through
    # two Adam parameter groups is impossible inside one tensor. A post-step
    # row-specific Adam update scales the actual Adam update (not the gradient,
    # which Adam would normalize away), while keeping all old rows conservative.
    trainables=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainables,lr=args.learning_rate,weight_decay=0.01)
    new_params=(model.decoder.embedding.weight,model.joint.head.weight,model.joint.head.bias)
    best_score=math.inf;best_state=None;best_step=None;history=[]
    progress_path=OUTPUT/'training_history.json'
    print(f'trainable params {sum(p.numel() for p in trainables):,}; Turkish {len(tr)}; replay {len(replay)}',flush=True)
    write_json(OUTPUT/'run_config.json',{**vars(args),'base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b',
        'transformers_commit':'536ecc007387a50e77603bb5d92100e9b07514cc','encoder_frozen':True,
        'batchnorm_stats_frozen':True,'replay_fraction':0.5,'turkish_clips_used':len(tr),
        'replay_clips_used':len(replay),'trainable_parameters':sum(p.numel() for p in trainables),
        'replay_new_token_exclusions':replay_new_token_exclusions,
        'lexical_roundtrip_exclusions':roundtrip_exclusions,
        'hardware':torch.cuda.get_device_name(0),'training_stream':'alternating Turkish and uniformly selected retention language',
        'confidence_calibration':'separate dev-only temperatures, no transcript changes',
        'retention_gate':'each dev WER <= base + max(1 percentage point,10% relative base WER)',
        'cache_policy':'base frozen encoder features only; labels must match extended tokenizer audit'} )
    train_mode(model)
    start=time.perf_counter()
    for step in range(1,args.steps+1):
        optimizer.zero_grad(set_to_none=True);tdt_sum=kl_sum=0.0
        for micro in range(args.accumulation):
            if micro%2==0:
                row=random.choice(tr)
            else:
                lang=random.choice(list(by_lang));row=random.choice(by_lang[lang])
            item=torch.load(row['cache_path'],map_location='cpu',weights_only=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out,features,decoder_ids=forward_cached(model,item)
                kl=teacher_kl(model,teacher,features,decoder_ids,out.logits,old_size) if row['language']!='tr' else torch.zeros((),device='cuda')
                loss=(out.loss+args.replay_kl*kl)/args.accumulation
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step} {row["source_id"]}')
            loss.backward();tdt_sum+=float(out.loss.detach());kl_sum+=float(kl.detach())
            del out,features,decoder_ids,loss
        norm=torch.nn.utils.clip_grad_norm_(trainables,1.0,error_if_nonfinite=True)
        # Accelerate learning for the six new rows. Only these updates are scaled.
        previous=[p[old_size:model.config.vocab_size].detach().clone() for p in new_params]
        optimizer.step()
        with torch.no_grad():
            for p,old in zip(new_params,previous):
                rows=p[old_size:model.config.vocab_size]
                rows.copy_(old+args.new_token_lr_multiplier*(rows-old))
        record={'step':step,'tdt_loss':tdt_sum/args.accumulation,'replay_kl':kl_sum/args.accumulation,
                'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-start,
                'peak_allocated_vram_gb':torch.cuda.max_memory_allocated()/1e9}
        history.append(record)
        if step%10==0:print(json.dumps(record),flush=True);write_json(progress_path,history)
        if step%args.eval_every==0 or step==args.steps:
            dev=evaluate(model,processor,f'{args.run_id}_step_{step}',splits=('validation',))
            deltas={lang:dev[f'validation_{lang}']['asr']['wer']-baseline[f'validation_{lang}']['asr']['wer'] for lang in by_lang}
            passed=all(deltas[lang]<=max(.01,.1*baseline[f'validation_{lang}']['asr']['wer']) for lang in by_lang)
            tr_score=dev['validation_tr']['asr']['wer']
            record['validation']={'turkish_wer':tr_score,'retention_wer_deltas':deltas,'retention_gate_passed':passed}
            if passed and tr_score<best_score and tr_score<baseline['validation_tr']['asr']['wer']:
                best_score=tr_score;best_state=checkpoint_state(model);best_step=step
                torch.save({'step':step,'weights':best_state},ROOT/'work/best_heads.pt')
            print(f'dev step{step} Turkish WER {tr_score:.4f}; retention passed {passed}; best {best_step}',flush=True)
            write_json(progress_path,history);train_mode(model)
    if best_state is None:
        # Keep the candidate for investigation, never label a failed retention run
        # as the recommended fine-tune.
        torch.save({'step':args.steps,'weights':checkpoint_state(model)},ROOT/'work/rejected_heads.pt')
        write_json(OUTPUT/'selection.json',{'accepted':False,'reason':'No checkpoint improved Turkish and passed all development retention gates.'})
        return
    model.load_state_dict(best_state,strict=False);model.eval()
    path=OUTPUT/'checkpoint'
    model.save_pretrained(path);processor.save_pretrained(path)
    write_json(OUTPUT/'selection.json',{'accepted':True,'selected_step':best_step,'turkish_validation_wer':best_score,
        'selection_split':'FLEURS validation only','checkpoint':str(path),'pilot':True})
    adapted=evaluate(model,processor,f'{args.run_id}_adapted')
    comparisons={}
    for lang in LANGUAGES:
        before=read_jsonl(OUTPUT/f'results/base_v3/test_{lang}.jsonl')
        after=read_jsonl(OUTPUT/f'results/{args.run_id}_adapted/test_{lang}.jsonl')
        ids=[r['source_id'] for r in before]
        after_by={r['source_id']:r for r in after};after=[after_by[key] for key in ids]
        comparisons[lang]=paired_bootstrap_delta([r['reference'] for r in before],
            [r['hypothesis'] for r in before],[r['hypothesis'] for r in after],language=lang,
            groups=[r.get('sentence_id') for r in before])
    write_json(OUTPUT/'benchmark_comparison.json',comparisons)
    write_json(OUTPUT/'confidence_calibration.json',{lang:adapted[f'calibration_{lang}'] for lang in LANGUAGES})
    write_json(OUTPUT/'adapted_results_path.json',{'tag':f'{args.run_id}_adapted','summary':str(OUTPUT/f'results/{args.run_id}_adapted/summary.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--eval-every',type=int,default=40)
    p.add_argument('--train-limit',type=int,default=2000);p.add_argument('--max-tokens',type=int,default=180)
    p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--new-token-lr-multiplier',type=float,default=100.0)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--replay-kl',type=float,default=.5)
    p.add_argument('--seed',type=int,default=20261010)
    p.add_argument('--run-id',default='pilot_'+uuid.uuid4().hex[:8])
    args=p.parse_args()
    if args.accumulation<2 or args.accumulation%2:raise ValueError('50:50 replay needs even accumulation >=2')
    run(args)
