"""Fair isolated GPU throughput/gradient profile of the next training path.

Every invocation starts from the same trained pilot snapshot and processes the
same four TRAIN clips. No experiment checkpoint or original helper is changed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from runtime import ROOT, OUTPUT, load_model, inputs_for
from extend_vocab import extend_processor_and_model
from encoder_lora import install_encoder_lora
from train_lora import checkpoint_encoder_blocks, preservation_losses
from efficient_tdt import chunked_tdt_loss as original_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(args):
    torch.manual_seed(20261010)
    torch.cuda.manual_seed_all(20261010)
    model, processor = load_model()
    old_size = model.config.vocab_size
    teacher = {name: copy.deepcopy(getattr(model, name)).eval().requires_grad_(False)
               for name in ('encoder_projector', 'decoder', 'joint')}
    extend_processor_and_model(processor, model)
    install_encoder_lora(model, rank=8, alpha=16)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state['weights']) != expected:
        raise ValueError('Profiler requires the exact pilot trainable layout.')
    model.load_state_dict(state['weights'], strict=False)
    if not args.no_encoder_checkpoint:
        checkpoint_encoder_blocks(model)
    model.train()
    model.encoder.eval()
    return model, processor, teacher, old_size, state


def examples(args):
    assessment = json.loads((OUTPUT / 'batching_assessment.json').read_text(encoding='utf-8'))
    q = assessment['representative_train_quartets'][args.quartet]
    selected = [q['tr'][0], q['replay'][0], q['tr'][1], q['replay'][1]]
    manifest_rows = {}
    for path in {r['manifest'] for r in selected}:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            manifest_rows[row['source_id']] = row
    rows, items = [], []
    for specification in selected:
        row = dict(manifest_rows[specification['source_id']])
        row['cache_path'] = specification['cache_path']
        if row['dataset_split'] != 'train':
            raise ValueError('Profiling must use TRAIN only.')
        if digest(row['audio_filepath']) != specification['wav_sha256']:
            raise ValueError('Profile WAV changed.')
        item = torch.load(row['cache_path'], map_location='cpu', weights_only=True)
        if item['source_id'] != row['source_id']:
            raise ValueError('Cache provenance differs.')
        rows.append(row)
        items.append(item)
    return rows, items


def original_objective(model, processor, teacher, old_size, row, item, chunk, inputs=None, debug=False, use_autocast=True):
    labels = item['labels'][None].cuda().long()
    ids = torch.cat([torch.full((1, 1), model.config.blank_token_id, device='cuda', dtype=torch.long), labels], dim=1)
    with torch.autocast('cuda', dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext():
        encoded = model.encoder(**(inputs if inputs is not None else inputs_for(processor, row)))
        features = encoded.last_hidden_state[:, :int(encoded.attention_mask.sum())]
        ep, dp = model.encoder_projector(features), model.decoder(ids)
        tdt = original_loss(model.joint, ep, dp, labels, model.config.vocab_size,
                            model.config.blank_token_id, model.config.durations,
                            logit_lengths=encoded.attention_mask.sum(-1).long(), chunk_size=chunk)
        kl = duration = feature = torch.zeros((), device='cuda')
        if row['language'] != 'tr':
            base = item['features'][None].cuda().float()
            if base.shape != features.shape:
                raise ValueError('Student and teacher lengths differ.')
            kl, duration = preservation_losses(model, teacher, base, ep, dp, ids, old_size)
            feature = (1 - torch.nn.functional.cosine_similarity(features.float(), base, dim=-1)).mean()
        loss = tdt + 5 * kl + .5 * duration + feature
    return loss, [tdt, kl, duration, feature], features.detach().cpu() if debug else None


def run(args):
    from batched_lora import collate_training_batch, batch_objective
    if args.tdt == 'v2':
        from efficient_tdt_v2 import chunked_tdt_loss as loss_fn
    else:
        loss_fn = original_loss
    code_names = ('profile_batched_training.py', 'batched_lora.py', 'efficient_tdt.py', 'efficient_tdt_v2.py',
                  'runtime.py', 'encoder_lora.py', 'extend_vocab.py', 'train_lora.py')
    code_start = {name: digest(OUTPUT / name) for name in code_names if (OUTPUT / name).exists()}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, items = examples(args)
    model, processor, teacher, old_size, snapshot = initialize(args)
    if args.batch_size == 1:
        groups = [[0], [1], [2], [3]]
    elif args.batch_size == 2:
        groups = [[0, 2], [1, 3]]
    else:
        groups = [[0, 1, 2, 3]]
    def collate(group, device):
        return collate_training_batch(processor, [rows[i] for i in group], [items[i] for i in group],
                                      blank_id=model.config.blank_token_id, old_size=old_size, device=device)
    cpu_batches = [collate(group, 'cpu') for group in groups]
    def get_batch(i, prefetched):
        return cpu_batches[i].to('cuda') if prefetched else collate(groups[i], 'cuda')
    parameter_items = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    adapters = [p for n, p in parameter_items if n.endswith(('lora_A', 'lora_B'))]
    heads = [p for n, p in parameter_items if not n.endswith(('lora_A', 'lora_B'))]
    optimizer = torch.optim.AdamW([{'params': heads, 'lr': 2e-5, 'weight_decay': .01},
                                   {'params': adapters, 'lr': 3e-4, 'weight_decay': 0}])
    new_parameters = [model.decoder.embedding.weight, model.joint.head.weight, model.joint.head.bias]

    def backward(prefetched, original=False, audit=False):
        optimizer.zero_grad(set_to_none=True)
        components = [0., 0., 0., 0.]
        feature_debug = []
        if original:
            for i, (row, item) in enumerate(zip(rows, items)):
                inputs = get_batch(i, True).inputs if prefetched else None
                loss, values, feature = original_objective(model, processor, teacher, old_size, row, item, 16, inputs, audit,
                                                         use_autocast=args.precision == 'bf16')
                (loss / 4).backward()
                components = [a + float(v.detach()) / 4 for a, v in zip(components, values)]
                if audit:
                    feature_debug.append(feature)
        else:
            for i in range(len(groups)):
                batch = get_batch(i, prefetched)
                result = batch_objective(model, teacher, batch, old_size=old_size,
                                          replay_kl=5, duration_kl=.5, feature_distillation=1,
                                          chunk_size=args.chunk_size, tdt_loss_fn=loss_fn,
                                          tdt_host_lengths=args.tdt == 'v2', return_debug_features=audit,
                                          use_autocast=args.precision == 'bf16')
                (result.loss / len(groups)).backward()
                values = [result.tdt_loss, result.replay_kl, result.replay_duration_kl,
                          result.replay_feature_distillation]
                components = [a + float(v.detach()) / len(groups) for a, v in zip(components, values)]
                if audit:
                    for row_index, original_index in enumerate(groups[i]):
                        length = result.encoder_frames[row_index]
                        feature_debug.append((original_index, result.debug_encoder_features[row_index:row_index+1, :length].cpu()))
                del result, batch
        if audit:
            missing = [name for name, p in parameter_items if p.grad is None]
            bad = [name for name, p in parameter_items if p.grad is not None and not torch.isfinite(p.grad).all()]
            if missing or bad:
                raise ValueError({'missing': missing, 'nonfinite': bad})
        return components, feature_debug

    reference_path = args.reference_path or ROOT / f'work/profile_gradients_{args.quartet}_{args.precision}_reference.pt'
    reference_identity = {'weights_sha256': digest(args.weights), 'precision': args.precision,
                          'source_ids': [r['source_id'] for r in rows],
                          'wav_sha256': [digest(r['audio_filepath']) for r in rows],
                          'cache_sha256': [digest(r['cache_path']) for r in rows],
                          'original_dependency_sha256': {n: code_start[n] for n in ('runtime.py','encoder_lora.py','extend_vocab.py','train_lora.py','efficient_tdt.py')}}
    if args.reference or args.reference_batched:
        if args.reference and (args.batch_size != 1 or args.no_encoder_checkpoint or args.tdt != 'original' or args.chunk_size != 16):
            raise ValueError('The original reference must use B1 and block checkpointing.')
        components, debug = backward(args.reference_batched, original=args.reference, audit=True)
        if args.reference_batched:
            debug = [value for _, value in sorted(debug, key=lambda item: item[0])]
        reference = {name: p.grad.detach().cpu().clone() for name, p in parameter_items}
        torch.save({'identity': reference_identity, 'reference_batch_size': args.batch_size, 'components': components,
                    'gradients': reference, 'encoder_features': debug}, reference_path)
        parity = {'reference_created': str(reference_path), 'passed': True, 'components': components}
    else:
        reference = torch.load(reference_path, map_location='cpu', weights_only=True)
        if reference['identity'] != reference_identity:
            raise ValueError('Gradient reference checkpoint differs.')
        components, feature_debug = backward(True, audit=True)
        norm2 = delta2 = dot = candidate2 = 0.
        per_parameter = []
        for name, p in parameter_items:
            a, b = reference['gradients'][name].double(), p.grad.detach().cpu().double()
            an, bn = float(a.square().sum()), float(b.square().sum())
            dn, ab = float((a - b).square().sum()), float((a * b).sum())
            norm2 += an
            candidate2 += bn
            delta2 += dn
            dot += ab
            per_parameter.append({'name': name, 'reference_norm': an**.5,
                                  'candidate_norm': bn**.5, 'delta_norm': dn**.5, 'dot': ab,
                                  'relative_l2_difference': (dn / max(an, 1e-20))**.5})
        relative = (delta2 / max(norm2, 1e-20))**.5
        cosine = dot / max((norm2 * candidate2)**.5, 1e-20)
        group_gradients = {}
        for group_name, adapter_group in (('encoder_lora', True), ('heads', False)):
            selected = [p for p in per_parameter if p['name'].endswith(('lora_A','lora_B')) == adapter_group]
            gn = sum(p['reference_norm']**2 for p in selected)
            cn = sum(p['candidate_norm']**2 for p in selected)
            group_gradients[group_name] = {'relative_l2_difference': (sum(p['delta_norm']**2 for p in selected)/max(gn,1e-20))**.5,
                                          'cosine_similarity': sum(p['dot'] for p in selected)/max((gn*cn)**.5,1e-20)}
        component_deltas = [abs(a - b) for a, b in zip(reference['components'], components)]
        # BF16 matrix shape changes permit small rounding differences. Objective
        # semantic correctness is separately checked in the CPU batch audit.
        feature_differences = []
        for index, candidate in sorted(feature_debug, key=lambda item: item[0]):
            original = reference['encoder_features'][index]
            if candidate.shape != original.shape:
                raise ValueError('Batched and original valid encoder shape differs.')
            feature_differences.append(float((candidate.float()-original.float()).norm() / original.float().norm().clamp_min(1e-12)))
        passed = (relative < .05 and cosine > .995 and component_deltas[0] < .02
                  and max(component_deltas[1:]) < .01 and max(feature_differences) < .02)
        parity = {'passed': passed, 'components': components, 'reference_components': reference['components'],
                  'component_absolute_deltas': component_deltas, 'gradient_relative_l2_difference': relative,
                  'gradient_cosine_similarity': cosine, 'per_parameter': per_parameter,
                  'gradient_groups': group_gradients,
                  'encoder_feature_relative_l2_differences': feature_differences,
                  'tolerances': {'gradient_relative_l2': .05, 'gradient_cosine': .995,
                                 'tdt_loss_absolute': .02, 'preservation_loss_absolute': .01,
                                 'encoder_feature_relative_l2': .02}}
        if not passed:
            print(json.dumps({'parity_failed': parity}), flush=True)
    optimizer.zero_grad(set_to_none=True)

    def update(prefetched):
        backward(prefetched, original=args.reference)
        torch.nn.utils.clip_grad_norm_([p for _, p in parameter_items], 1., error_if_nonfinite=True)
        previous = [p[old_size:model.config.vocab_size].detach().clone() for p in new_parameters]
        optimizer.step()
        with torch.no_grad():
            for p, old in zip(new_parameters, previous):
                new = p[old_size:model.config.vocab_size]
                new.copy_(old + 15 * (new - old))

    timing = {}
    if parity['passed'] and not args.parity_only:
        for prefetched in (True, False):
            # Reset model and optimizer to the identical starting state for each
            # timing category. No timings contain development or inference.
            model.load_state_dict(snapshot['weights'], strict=False)
            optimizer.state.clear()
            for _ in range(args.warmup):
                update(prefetched)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            seconds = []
            for index in range(args.updates):
                torch.cuda.synchronize()
                start = time.perf_counter()
                update(prefetched)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
                print(json.dumps({'timing': 'prefetched' if prefetched else 'end_to_end',
                                  'update': index + 1, 'seconds': seconds[-1]}), flush=True)
            audio_seconds = sum(row['duration'] for row in rows) * args.updates
            sorted_seconds = sorted(seconds)
            timing['prefetched_cpu_inputs' if prefetched else 'end_to_end'] = {
                'updates': args.updates, 'warmup_updates': args.warmup, 'seconds': seconds,
                'median_seconds_per_update': statistics.median(seconds),
                'p90_seconds_per_update': sorted_seconds[min(len(seconds)-1, int(.9 * len(seconds)))],
                'audio_min_per_wall_min': audio_seconds / sum(seconds),
                'clips_per_second': 4 * args.updates / sum(seconds),
                'peak_allocated_vram_gb': torch.cuda.max_memory_allocated() / 1e9,
                'peak_reserved_vram_gb': torch.cuda.max_memory_reserved() / 1e9}
    report = {'status': 'passed' if parity['passed'] else 'parity_failed', 'quartet': args.quartet,
              'batch_size': args.batch_size, 'accumulation': len(groups), 'tdt_implementation': args.tdt,
              'chunk_size': args.chunk_size, 'encoder_checkpointing': not args.no_encoder_checkpoint,
              'reference_uses_frozen_original_single_clip_objective': args.reference,
              'reference_batched': args.reference_batched, 'reference_path': str(reference_path), 'precision': args.precision,
              'weights_sha256': digest(args.weights), 'source_ids': [r['source_id'] for r in rows],
              'audio_seconds_per_update': sum(r['duration'] for r in rows),
              'parity': parity, 'timing': timing, 'hardware': torch.cuda.get_device_name(0),
              'code_sha256': code_start}
    if any(digest(OUTPUT / name) != sha for name, sha in code_start.items()):
        raise ValueError('A source dependency changed while this profile was running.')
    target = OUTPUT / args.output
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'saved': str(target), 'status': report['status'], 'timing': timing}), flush=True)
    if not parity['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--tdt', choices=('original', 'v2'), default='v2')
    parser.add_argument('--quartet', default='median')
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--no-encoder-checkpoint', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--reference-batched', action='store_true')
    parser.add_argument('--reference-path', type=Path)
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--precision', choices=('bf16','fp32'), default='bf16')
    parser.add_argument('--weights', type=Path, default=ROOT / 'work/lora_20261010_01_best.pt')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--output', default='batch_profile.json')
    run(parser.parse_args())
