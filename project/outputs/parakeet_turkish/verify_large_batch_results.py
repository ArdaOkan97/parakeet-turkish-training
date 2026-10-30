"""CPU-only independent audit of matched effective16 GPU batch profiles.

Numerical equivalence and measured throughput are reported separately.  No
model, inference, or CUDA execution occurs here.  Correctness comparisons use
the saved tensors rather than trusting a profiler's acceptance flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import csv
from pathlib import Path
import statistics
from datetime import datetime, timezone

import torch
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COEFFICIENTS = (1., 5., .5, 1.)
COMPONENTS = ('tdt', 'token_kl', 'duration_kl', 'feature_distillation')


class Checks:
    def __init__(self):
        self.items = []

    def require(self, condition, label):
        self.items.append({'name': label, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(label)

    def near(self, actual, expected, label, *, rtol=2e-7, atol=1e-9):
        self.require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                     and math.isfinite(actual) and math.isfinite(expected)
                     and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def resolved(value, check, label, *, exists=True):
    path = Path(value).resolve()
    check.require(path.is_relative_to(ROOT.resolve()), label + ': path within workspace')
    if exists:
        check.require(path.is_file(), label + ': file exists')
    return path


def tensor_finite(value, check, label):
    check.require(isinstance(value, torch.Tensor) and value.device.type == 'cpu', label + ': saved CPU tensor')
    check.require(bool(torch.isfinite(value).all()), label + ': finite')


def norm_stats(norm2, candidate2, delta2, dot):
    if norm2 == candidate2 == delta2 == 0:
        relative, cosine = 0., 1.
    else:
        relative = math.sqrt(delta2 / max(norm2, 1e-30))
        cosine = dot / max(math.sqrt(norm2 * candidate2), 1e-30)
    return {'reference_norm': math.sqrt(norm2), 'candidate_norm': math.sqrt(candidate2),
            'delta_norm': math.sqrt(delta2), 'relative_l2_difference': relative,
            'cosine_similarity': cosine}


def tensor_comparison(reference, candidate, check, label):
    """Recompute full/global/group sufficient statistics in CPU float64."""
    check.require(set(reference) == set(candidate) and bool(reference), label + ': parameter layout')
    total = [0., 0., 0., 0.]
    groups = {'encoder_lora': [0., 0., 0., 0.], 'heads': [0., 0., 0., 0.]}
    parameters = []
    for name in sorted(reference):
        a, b = reference[name], candidate[name]
        tensor_finite(a, check, label + ': reference ' + name)
        tensor_finite(b, check, label + ': candidate ' + name)
        check.require(a.shape == b.shape, label + ': shape ' + name)
        equal = torch.equal(a, b)
        a, b = a.double(), b.double()
        values = [float(a.square().sum()), float(b.square().sum()),
                  float((a-b).square().sum()), float((a*b).sum())]
        group = 'encoder_lora' if name.endswith(('lora_A', 'lora_B')) else 'heads'
        total = [left+right for left, right in zip(total, values)]
        groups[group] = [left+right for left, right in zip(groups[group], values)]
        parameters.append({'name': name, 'elements': a.numel(), 'bitwise_equal': equal,
                           **norm_stats(*values)})
    return {'global_gradient': norm_stats(*total),
            'gradient_groups': {name: norm_stats(*value) for name, value in groups.items()},
            'per_parameter': parameters, 'trainable_tensors': len(parameters),
            'trainable_elements': sum(value['elements'] for value in parameters)}


def correctness_comparison(reference, candidate, check, label, *, same_shape=False,
                           allow_precision_difference=False, include_weighted_bound=True):
    left, right = dict(reference['identity']), dict(candidate['identity'])
    if allow_precision_difference:
        left.pop('precision', None)
        right.pop('precision', None)
    check.require(left == right, label + ': immutable data/checkpoint/precision identity')
    value = tensor_comparison(reference['gradients'], candidate['gradients'], check, label)
    check.require(len(reference['encoder_features']) == len(candidate['encoder_features']) == 16,
                  label + ': all16 valid encoder features')
    features = []
    for index, (a, b) in enumerate(zip(reference['encoder_features'], candidate['encoder_features'])):
        tensor_finite(a, check, label + ': reference feature ' + str(index))
        tensor_finite(b, check, label + ': candidate feature ' + str(index))
        check.require(a.shape == b.shape, label + ': valid feature shape ' + str(index))
        a64, b64 = a.double(), b.double()
        features.append({'source_id': reference['identity']['source_ids'][index],
                         'shape': list(a.shape), 'bitwise_equal': torch.equal(a, b),
                         'relative_l2_difference': float((a64-b64).norm()/a64.norm().clamp_min(1e-30)),
                         'max_absolute_difference': float((a64-b64).abs().max())})
    left_values, right_values = reference['components'], candidate['components']
    check.require(len(left_values) == len(right_values) == 4,
                  label + ': four mean objective components')
    check.require(all(math.isfinite(float(x)) for x in list(left_values)+list(right_values)),
                  label + ': finite objective components')
    deltas = {name: abs(float(a)-float(b)) for name, a, b in zip(COMPONENTS, left_values, right_values)}
    value.update({'encoder_features': features, 'component_absolute_deltas': deltas,
                  'weighted_objective_absolute_delta': abs(math.fsum(c*float(a) for c,a in zip(COEFFICIENTS,left_values))
                      - math.fsum(c*float(a) for c,a in zip(COEFFICIENTS,right_values))),
                  'reference_components': dict(zip(COMPONENTS,map(float,left_values))),
                  'candidate_components': dict(zip(COMPONENTS,map(float,right_values)))})
    limits = {'gradient_relative_l2': .0005 if same_shape else .002,
              'gradient_cosine_min': .9999995 if same_shape else .99999,
              'feature_relative_l2': 1e-4, 'component_absolute_delta': 1e-4,
              'weighted_objective_absolute_delta': 1e-4}
    decisions = {name + '_relative': stats['relative_l2_difference'] < limits['gradient_relative_l2']
                 for name,stats in {'global':value['global_gradient'],**value['gradient_groups']}.items()}
    decisions.update({name + '_cosine': stats['cosine_similarity'] > limits['gradient_cosine_min']
                      for name,stats in {'global':value['global_gradient'],**value['gradient_groups']}.items()})
    decisions['feature_relative'] = all(x['relative_l2_difference'] < limits['feature_relative_l2'] for x in features)
    decisions['components'] = max(deltas.values()) < limits['component_absolute_delta']
    if include_weighted_bound:
        decisions['weighted_objective'] = value['weighted_objective_absolute_delta'] < limits['weighted_objective_absolute_delta']
    else:
        limits['weighted_objective_absolute_delta'] = None
    if same_shape:
        decisions['features_bitwise'] = all(x['bitwise_equal'] for x in features)
    value['strict_gate'] = {'passed': all(decisions.values()), 'limits': limits,
                            'checks': decisions, 'same_shape': same_shape}
    return value


def timing_audit(value, audio, check, label, *, minimum_updates=10, minimum_warmup=2):
    count, seconds = value['updates'], value['seconds']
    check.require(type(count) is int and count >= minimum_updates and len(seconds) == count,
                  label + ': complete measured update count')
    check.require(type(value['warmup_updates']) is int and value['warmup_updates'] >= minimum_warmup,
                  label + ': excluded warmup updates')
    check.require(all(isinstance(x,(int,float)) and not isinstance(x,bool)
                      and math.isfinite(x) and x>0 for x in seconds), label + ': positive finite seconds')
    total = math.fsum(seconds)
    ordered = sorted(seconds)
    expected = {'median_seconds_per_update': statistics.median(seconds),
                'p90_seconds_per_update': ordered[min(count-1,int(.9*count))],
                'audio_min_per_wall_min': audio*count/total,
                'clips_per_second': 16*count/total}
    for name, amount in expected.items():
        if name == 'p90_seconds_per_update' and name not in value:
            continue  # Older serialization: derived here, never an observed omission claim.
        check.near(value[name], amount, label + ': recomputed ' + name)
    allocated, reserved = value['peak_allocated_vram_gb'], value['peak_reserved_vram_gb']
    check.require(math.isfinite(allocated) and math.isfinite(reserved) and 0<allocated<=reserved+1e-9,
                  label + ': bounded allocated/reserved peak memory')
    return {**expected, 'updates':count, 'warmup_updates':value['warmup_updates'],
            'total_measured_seconds':total, 'peak_allocated_vram_gb':allocated,
            'peak_reserved_vram_gb':reserved}


def adam_update_audit(evidence, snapshot, check, label):
    """Reconstruct the fresh clipped AdamW step, then the bounded row boost."""
    update = evidence['adam_update']
    old_size, vocab_size = update['old_size'], update['vocab_size']
    check.require(type(old_size) is int and type(vocab_size) is int and 0 < old_size < vocab_size,
                  label + ': ordered original/extended token bounds')
    check.require(update['optimizer'] == 'AdamW' and update['initial_state_empty'] is True,
                  label + ': fresh AdamW state')
    check.require(update['row_multiplier'] == 15. and update['max_grad_norm'] == 1.,
                  label + ': exact clipping and appended-token update policy')
    check.require(set(evidence['gradients']) == set(snapshot['weights']),
                  label + ': full checkpoint gradient layout for clipping')
    parameters = []
    for name in sorted(evidence['gradients']):
        gradient = evidence['gradients'][name]
        tensor_finite(gradient, check, label + ': Adam input ' + name)
        check.require(gradient.dtype == snapshot['weights'][name].dtype == torch.float32,
                      label + ': FP32 parameter and gradient storage ' + name)
        check.require(gradient.shape == snapshot['weights'][name].shape,
                      label + ': checkpoint gradient dimensions ' + name)
        parameter = torch.nn.Parameter(snapshot['weights'][name].detach().clone())
        parameter.grad = gradient.detach().clone()
        parameters.append((name, parameter))
    optimizer = torch.optim.AdamW([p for _,p in parameters], lr=update['lr'], betas=tuple(update['betas']),
                                  eps=update['eps'], weight_decay=update['weight_decay'], foreach=False)
    # CPU vector_norm FP32 can accumulate visibly more reduction error than
    # CUDA on the large vocabulary matrices. Sum squared saved gradients in
    # float64, then reproduce CUDA's FP32 scalar coefficient semantics.
    total_norm = math.sqrt(math.fsum(float(p.grad.double().square().sum()) for _,p in parameters))
    check.near(update['unclipped_global_norm'], total_norm, label + ': global clip norm', rtol=3e-6, atol=3e-6)
    coefficient=(torch.tensor(update['max_grad_norm'],dtype=torch.float32)
                 /(torch.tensor(total_norm,dtype=torch.float32)+1e-6)).clamp(max=1.)
    for _,parameter in parameters:
        parameter.grad.mul_(coefficient)
    optimizer.step()
    expected_names = {'decoder.embedding.weight', 'joint.head.weight', 'joint.head.bias'}
    # Parameter spelling comes from the pinned model, rather than accepting any
    # arbitrary tensors with a first dimension equal to vocabulary size.
    check.require(expected_names.issubset(snapshot['weights']) and set(update['parameters']) == expected_names,
                  label + ': exactly embedding/joint weight/joint bias row audits')
    results = {}
    for name, parameter in parameters:
        if name not in expected_names:
            continue
        initial = snapshot['weights'][name]
        recorded = update['parameters'][name]
        if 'unclipped_gradient' in recorded:
            check.require(torch.equal(recorded['unclipped_gradient'],evidence['gradients'][name]),
                          label+': exact saved unclipped gradient '+name)
            check.require(torch.allclose(recorded['clipped_gradient'],parameter.grad,rtol=3e-5,atol=1e-7),
                          label+': independent global-gradient clipping '+name)
        if 'exp_avg' in recorded:
            state=optimizer.state[parameter]
            check.require(recorded['optimizer_step']==1.,label+': fresh first Adam step '+name)
            for key in ('exp_avg','exp_avg_sq'):
                tensor_finite(recorded[key],check,label+': '+key+' '+name)
                check.require(torch.allclose(recorded[key],state[key],rtol=3e-5,atol=1e-10),
                              label+': independent fresh Adam moment '+key+' '+name)
        before_boost = parameter.detach().clone()
        boosted = before_boost.clone()
        boosted[old_size:vocab_size] = (initial[old_size:vocab_size]
                                       + 15.*(before_boost[old_size:vocab_size]-initial[old_size:vocab_size]))
        for key, expected in (('before',initial), ('after_adam',before_boost), ('after_multiplier',boosted)):
            actual = recorded[key]
            tensor_finite(actual,check,label+': '+name+' '+key)
            check.require(actual.shape == expected.shape,label+': '+name+' '+key+' shape')
            check.require(torch.allclose(actual,expected,rtol=3e-5,atol=3e-6),
                          label+': CPU reconstruction '+name+' '+key)
        check.require(torch.equal(recorded['before'],initial),label+': exact initial row tensor '+name)
        # These exact identities catch accidental multiplier application to
        # old tokens or to the five appended output duration rows.
        check.require(torch.equal(recorded['after_multiplier'][:old_size],recorded['after_adam'][:old_size]),
                      label+': original token rows unscaled '+name)
        if initial.shape[0] > vocab_size:
            check.require(initial.shape[0] == vocab_size+5,label+': exact five duration rows '+name)
            check.require(torch.equal(recorded['after_multiplier'][vocab_size:],recorded['after_adam'][vocab_size:]),
                          label+': duration rows unscaled '+name)
        expected_boost = (recorded['before'][old_size:vocab_size]
                          + 15.*(recorded['after_adam'][old_size:vocab_size]-recorded['before'][old_size:vocab_size]))
        check.require(torch.equal(recorded['after_multiplier'][old_size:vocab_size],expected_boost),
                      label+': exact appended-token multiplier '+name)
        for alias,target in (('after_row_scaling','after_multiplier'),('after_adam_before_row_scaling','after_adam')):
            if alias in recorded:
                check.require(torch.equal(recorded[alias],recorded[target]),label+': identical serialized Adam alias '+alias+' '+name)
        results[name] = {'original_rows':old_size,'new_token_rows':vocab_size-old_size,
                         'duration_rows':initial.shape[0]-vocab_size,
                         'max_cpu_adam_absolute_difference':float((recorded['after_adam']-before_boost).abs().max()),
                         'max_cpu_boosted_absolute_difference':float((recorded['after_multiplier']-boosted).abs().max())}
    return {'unclipped_global_norm':total_norm,'row_multiplier':15.,'parameters':results}


def training_inputs(pool_path, source_ids, check, label):
    """Check all16 TRAIN recordings/cache entries against immutable manifests."""
    pool_path = resolved(pool_path,check,label+': pool')
    pool = read(pool_path)
    pool_rows = pool['turkish'] + pool['replay']
    mapping = {row['source_id']:row for row in pool_rows}
    check.require(len(mapping) == len(pool_rows),label+': unique pool sources')
    check.require(len(source_ids) == len(set(source_ids)) == 16,label+': exactly16 unique recordings')
    check.require(all(sid in mapping for sid in source_ids),label+': all recordings in frozen TRAIN pool')
    selected = [mapping[sid] for sid in source_ids]
    check.require(sum(row['language']=='tr' for row in selected) == 8,label+': eight Turkish recordings')
    check.require(sum(row['language']!='tr' for row in selected) == 8,label+': eight replay recordings')
    check.require(all(row['dataset_split']=='train' for row in selected),label+': TRAIN only')
    selected_ids = set(source_ids)
    accepted_rows = {}
    heldout_sentences=set();heldout_audio=set();heldout_text=set()
    for manifest_name,expected in pool['input_sha256'].items():
        path = resolved(manifest_name,check,label+': input manifest')
        check.require(sha(path) == expected,label+': immutable input SHA '+path.name)
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['dataset_split']!='train':
                if row.get('sentence_id') is not None:
                    heldout_sentences.add(row['sentence_id'])
                heldout_audio.update(row[key] for key in ('audio_sha256','wav_sha256') if row.get(key))
                heldout_text.add((row['language'],row['reference_normalized']))
            if row['source_id'] not in selected_ids:
                continue
            check.require(row['dataset_split'] == 'train',label+': selected source excluded from heldout '+row['source_id'])
            check.require(row['source_id'] not in accepted_rows,label+': unique selected manifest source '+row['source_id'])
            accepted_rows[row['source_id']] = row
    check.require(set(accepted_rows) == selected_ids,label+': all selected sources have TRAIN manifest provenance')
    items = []
    for row in selected:
        sid = row['source_id']
        manifest = accepted_rows[sid]
        for key in ('language','text','dataset_split','duration','wav_sha256','audio_sha256','audio_filepath'):
            check.require(row[key] == manifest[key],label+': exact TRAIN '+key+' '+sid)
        check.require((row.get('sentence_id') is None or row['sentence_id'] not in heldout_sentences)
                      and not any(row.get(key) in heldout_audio for key in ('audio_sha256','wav_sha256'))
                      and (row['language'],row['reference_normalized']) not in heldout_text,
                      label+': excluded heldout sentence/audio/reference aliases '+sid)
        wav = resolved(row['audio_filepath'],check,label+': WAV')
        check.require(sha(wav) == row['wav_sha256'],label+': actual WAV SHA '+sid)
        cache_path = resolved(row['cache_path'],check,label+': cache')
        item = torch.load(cache_path,map_location='cpu',weights_only=True)
        check.require(item['source_id'] == sid,label+': cache source '+sid)
        tensor_finite(item['features'],check,label+': cache features '+sid)
        labels = item['labels']
        check.require(labels.ndim == 1 and labels.dtype == torch.int64 and labels.numel() > 0
                      and bool((labels >= 0).all()),label+': positive target labels '+sid)
        check.require(item['features'].ndim == 2 and item['features'].shape[1] == 1024
                      and item['features'].shape[0] == row['encoder_frames'],label+': true cached frame length '+sid)
        check.require(labels.numel() == row['target_tokens'],label+': true cached target length '+sid)
        check.require(labels.max() < (8199 if row['language']=='tr' else 8193),label+': valid teacher/tokenizer target IDs '+sid)
        wav_info = sf.info(wav)
        check.require(wav_info.samplerate == 16000 and wav_info.channels == 1,label+': mono16k prepared WAV '+sid)
        check.near(wav_info.frames/16000,row['duration'],label+': actual PCM duration '+sid,atol=1e-5)
        items.append({'labels':labels,'features':item['features'],'frame_length':len(item['features']),
                      'input_frames':wav_info.frames//160,'target_length':len(labels),
                      'cache_sha256':sha(cache_path),'wav_sha256':sha(wav),
                      'labels_sha256':hashlib.sha256(labels.numpy().tobytes()).hexdigest()})
    return selected,items


def mask_audit(microbatches, rows, items, batch_size, check, label):
    """Reconstruct blank pads, right input masks and teacher/replay indexing."""
    check.require(batch_size in (1,2,4,8,16),label+': allowed physical batch')
    check.require(len(microbatches) == 16//batch_size,label+': exact effective16 accumulation')
    indexes = [i for batch in microbatches for i in batch['indices']]
    check.require(sorted(indexes) == list(range(16)),label+': once-only recordings')
    expected_groups = ([list(range(16))] if batch_size == 16 else
                       [[i] for i in range(16)] if batch_size == 1 else
                       [group for start in range(0,8,batch_size) for group in
                        (list(range(0,16,2))[start:start+batch_size],list(range(1,16,2))[start:start+batch_size])])
    check.require([batch['indices'] for batch in microbatches] == expected_groups,label+': exact balanced microbatch grouping')
    valid_lattice = padded_lattice = valid_frames = padded_frames = 0
    result = []
    for number,batch in enumerate(microbatches):
        ids = batch['indices']
        check.require(len(ids) == batch_size,label+': full physical batch '+str(number))
        ts = [items[i]['frame_length'] for i in ids]
        us = [items[i]['target_length'] for i in ids]
        replay = [j for j,i in enumerate(ids) if rows[i]['language']!='tr']
        check.require(batch['replay_indices'] == replay,label+': replay indexing '+str(number))
        frame_lengths = batch['frame_lengths']
        target_lengths = batch['target_lengths']
        check.require(frame_lengths.dtype == target_lengths.dtype == batch['labels'].dtype
                      == batch['decoder_input_ids'].dtype == torch.int64,label+': integer metadata/target storage '+str(number))
        check.require(torch.equal(frame_lengths,torch.tensor(ts,dtype=torch.int64)),label+': original T metadata '+str(number))
        check.require(torch.equal(target_lengths,torch.tensor(us,dtype=torch.int64)),label+': original U metadata '+str(number))
        expected = torch.full((batch_size,max(us)),8192,dtype=torch.int64)
        for j,index in enumerate(ids):
            expected[j,:us[j]] = items[index]['labels']
        check.require(torch.equal(batch['labels'],expected),label+': exact labels/blank padding '+str(number))
        decoder = torch.cat((torch.full((batch_size,1),8192,dtype=torch.int64),expected),dim=1)
        check.require(torch.equal(batch['decoder_input_ids'],decoder),label+': exact decoder prefix and labels '+str(number))
        check.require(batch['input_attention_mask'].dtype == torch.bool,label+': original boolean input mask '+str(number))
        mask = batch['input_attention_mask'].bool()
        check.require(mask.ndim == 2 and mask.shape[0] == batch_size,label+': input mask shape '+str(number))
        lengths = mask.sum(-1)
        expected_right = torch.arange(mask.shape[1])[None,:] < lengths[:,None]
        check.require(bool((lengths>0).all()) and torch.equal(mask,expected_right),label+': positive right-padded input mask '+str(number))
        # The exact processor convention is floor(samples/160), followed by
        # three stride2 encoder convolutions. Cache T is an independent check.
        expected_mel = torch.tensor([items[i]['input_frames'] for i in ids],dtype=torch.int64)
        check.require(torch.equal(lengths,expected_mel),label+': input frames from real audio duration '+str(number))
        check.require(torch.equal((lengths+7)//8,frame_lengths),label+': subsampling mask equals teacher T '+str(number))
        mel = batch['input_features']
        base = batch['base_features']
        tensor_finite(mel,check,label+': input mel '+str(number))
        tensor_finite(base,check,label+': padded teacher '+str(number))
        check.require(mel.shape[:2] == mask.shape and mel.shape[2] == 128,label+': mel dimensions '+str(number))
        check.require(base.shape == (batch_size,max(ts),1024),label+': padded teacher dimensions '+str(number))
        check.require(bool((mel.masked_select(~mask.unsqueeze(-1).expand_as(mel))==0).all()),label+': zero input mel padding '+str(number))
        for j,index in enumerate(ids):
            check.require(torch.equal(base[j,:ts[j]],items[index]['features'].float()),label+': exact original cached teacher '+str(index))
            check.require(bool((base[j,ts[j]:]==0).all()),label+': zero teacher padding '+str(index))
        cells = sum(t*(u+1) for t,u in zip(ts,us))
        padded = batch_size*max(ts)*(max(us)+1)
        valid_lattice += cells;padded_lattice += padded
        valid_frames += sum(ts);padded_frames += batch_size*max(ts)
        result.append({'indices':ids,'replay_indices':replay,'frame_lengths':ts,'target_lengths':us,
                       'valid_lattice_cells':cells,'padded_lattice_cells':padded,
                       'lattice_padding_factor':padded/cells,'frame_padding_factor':batch_size*max(ts)/sum(ts)})
    return {'microbatches':result,'lattice_padding_factor':padded_lattice/valid_lattice,
            'frame_padding_factor':padded_frames/valid_frames,'valid_lattice_cells':valid_lattice,
            'padded_lattice_cells':padded_lattice}


def state_digest(weights):
    digest=hashlib.sha256()
    for name,tensor in sorted(weights.items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selected_ids(pool,family):
    """Independent deterministic selection; preserves the original tie keys."""
    def ordered(rows):
        if family=='long':
            return sorted(rows,key=lambda row:(-row['encoder_frames']*(row['target_tokens']+1),
                                               hashlib.sha256(row['source_id'].encode()).hexdigest()))
        mt=statistics.median(row['encoder_frames'] for row in rows)
        mu=statistics.median(row['target_tokens'] for row in rows)
        return sorted(rows,key=lambda row:(abs(math.log(row['encoder_frames']/mt))+
            abs(math.log(row['target_tokens']/mu)),hashlib.sha256(row['source_id'].encode()).hexdigest()))
    tr=ordered(pool['turkish'])[:8]
    languages=('en','es','it','fr','de','en','es','it')
    cursors=dict.fromkeys(languages,0);replay=[]
    for language in languages:
        options=ordered([row for row in pool['replay'] if row['language']==language])
        replay.append(options[cursors[language]]);cursors[language]+=1
    return [row['source_id'] for pair in zip(tr,replay) for row in pair]


def compare_recorded(actual,expected,check,label):
    """Require recorded statistics to agree with independent saved-tensor sums."""
    for field in ('global_gradient','gradient_groups'):
        if field=='global_gradient':
            pairs=[(field,actual[field],expected[field])]
        else:
            check.require(set(actual[field])==set(expected[field]),label+': gradient group keys')
            pairs=[(name,actual[field][name],value) for name,value in expected[field].items()]
        for name,left,right in pairs:
            for key,value in right.items():
                check.near(left[key],value,label+': saved '+name+' '+key,atol=1e-10)
    for key in ('trainable_tensors','trainable_elements'):
        check.require(actual[key]==expected[key],label+': '+key)
    pmap={row['name']:row for row in actual['per_parameter']}
    check.require(len(pmap)==len(actual['per_parameter'])==len(expected['per_parameter']),label+': unique parameter statistics')
    for row in expected['per_parameter']:
        saved=pmap[row['name']]
        for key,value in row.items():
            if key in ('name','elements','bitwise_equal'):
                check.require(saved[key]==value,label+': parameter '+row['name']+' '+key)
            else:
                check.near(saved[key],value,label+': parameter '+row['name']+' '+key,atol=1e-10)
    for key,value in expected['component_absolute_deltas'].items():
        check.near(actual['component_absolute_deltas'][key],value,label+': component '+key)
    check.near(actual['weighted_objective_absolute_delta'],expected['weighted_objective_absolute_delta'],label+': weighted objective')
    check.require(len(actual['encoder_features'])==16,label+': saved16 feature comparisons')
    for saved,row in zip(actual['encoder_features'],expected['encoder_features']):
        for key,value in row.items():
            if key in ('source_id','shape','bitwise_equal'):
                check.require(saved[key]==value,label+': feature '+key)
            else:
                check.near(saved[key],value,label+': feature '+key)


def per_clip_audit(evidence,rows,check,label):
    values=evidence['per_clip']
    check.require(set(values)==set(COMPONENTS)|{'total'},label+': exact per-clip objective keys')
    check.require(all(len(value)==16 for value in values.values()),label+': per-clip16 terms')
    check.require(all(math.isfinite(float(x)) for value in values.values() for x in value),label+': finite per-clip terms')
    for index,row in enumerate(rows):
        expected=sum(c*float(values[key][index]) for c,key in zip(COEFFICIENTS,COMPONENTS))
        check.near(values['total'][index],expected,label+': weighted per-clip objective '+str(index),rtol=3e-6,atol=3e-6)
        if row['language']=='tr':
            check.require(all(values[key][index]==0. for key in COMPONENTS[1:]),label+': literal zero Turkish preservation '+str(index))
        for key in COMPONENTS:
            check.require(float(values[key][index])>=(-1e-5 if key!='tdt' else 0.),label+': bounded per-clip '+key+' '+str(index))
    for index,key in enumerate(COMPONENTS):
        check.near(float(evidence['components'][index]),math.fsum(values[key])/16,
                   label+': per-clip mean denominator16 '+key,rtol=3e-6,atol=3e-6)


def pause_audit(value,check,label):
    if value is None:
        return {'available':False,'scope':'Solo context is operator-declared; no saved cooperative pause evidence.'}
    path=resolved(value['path'],check,label+': pause manifest')
    check.require(sha(path)==value['sha256'],label+': pause manifest SHA at profile')
    pause=read(path)
    check.require(pause['status']=='paused_checkpoint_verified',label+': cooperative pause completed')
    check.require(sha(HERE/'run_experiment_loop.py')==pause['original_controller_sha256'],label+': frozen original controller')
    for stem in ('original_status','original_config_before_pause','paused_config','paused_history','paused_selection',
                 'original_status_after_pause','checkpoint'):
        item=resolved(pause[stem+'_path'],check,label+': '+stem)
        check.require(sha(item)==pause[stem+'_sha256'],label+': immutable pause '+stem)
    check.require(type(pause['paused_step']) is int and pause['paused_step']>=0 and pause['optimizer_tensors']==207,
                  label+': saved pause step and full optimizer layout')
    history=read(pause['paused_history_path'])
    check.require(len(history)==pause['paused_step'] and all(row['step']==i+1 for i,row in enumerate(history)),
                  label+': exact paused contiguous history')
    return {'available':True,'path':str(path),'sha256_at_profile':value['sha256'],
            'run_id':pause['run_id'],'paused_step':pause['paused_step'],'paused_at_utc':pause['paused_at_utc']}


def inventory_audit(value,check,label):
    check.require(type(value.get('self_pid')) is int and value['self_pid']>0,label+': profiler PID')
    check.require(value.get('returncode')==0 and 'stdout' in value,label+': successful saved driver inventory')
    entries=[]
    for fields in csv.reader(value['stdout'].splitlines()):
        if not fields:
            continue
        check.require(len(fields)>=3 and fields[0].strip().isdigit(),label+': parseable driver row')
        pid=int(fields[0].strip());name=fields[1].strip();memory=fields[2].strip()
        entries.append({'pid':pid,'name':name,'memory':memory})
        normalized=name.replace('\\','/').rsplit('/',1)[-1].lower()
        check.require(pid==value['self_pid'] or normalized not in ('python.exe','pythonw.exe'),
                      label+': no other Python CUDA compute process')
    return {'self_pid':value['self_pid'],'entries':entries,
            'scope':'Saved Windows driver inventory plus parent/operator solo declaration; graphics contexts may remain.'}


def metadata_audit(path,entry,check):
    path=resolved(path,check,'profile report')
    check.require(sha(path)==entry['report_sha256'],'exact matrix report SHA '+entry['case_id'])
    report=read(path);label=entry['case_id']
    check.require(label==path.stem,label+': report filename/case identity')
    check.require(report['family'] in ('typical','long') and report['mode'] in
                  ('fp32-reference','fp32-parity','bf16-reference','bf16-parity','timing')
                  and report['backend'] in ('math','default'),label+': finite supported profiling matrix')
    for key in ('case_id','family','mode','batch_size','precision','backend','status'):
        check.require(report[key]==entry[key],label+': matrix/report '+key)
    check.require(report['tdt_implementation']==entry['tdt'],label+': matrix/report TDT implementation')
    check.require(report['effective_batch']==16 and report['batch_size'] in (1,2,4,8,16),label+': effective16 only')
    if report['status'] in ('error','error_oom'):
        check.require(not report['timing'] and bool(report['error']),label+': retained failure has no speed claim')
        return report
    check.require(report['status'] in ('reference_saved','strict_parity_failed','measured','measured_with_numerical_gate_failure'),label+': recognized measured/reference status')
    required={'profile_large_batches.py','profile_batched_training.py','diagnose_batch_objective_math.py','batched_lora.py',
              'efficient_tdt.py','efficient_tdt_v2.py','runtime.py','encoder_lora.py','extend_vocab.py','train_lora.py',
              'train.py','train_lora_batched.py'}
    check.require(set(report['source_sha256'])==required,label+': full frozen source dictionary')
    for name,expected in report['source_sha256'].items():
        check.require(sha(HERE/name)==expected,label+': source SHA '+name)
    check.require(report['accumulation']==16//report['batch_size'] and report['encoder_checkpointing']
                  and report['chunk_size']==16,label+': exact accumulation/chunk/checkpoint settings')
    check.require(report['objective_coefficients']==dict(zip(COMPONENTS,COEFFICIENTS))
                  and report['turkish_preservation_terms_zero'] is True,label+': unchanged per-clip objective')
    check.require(report['gpu_context']=='solo' and report['adoption_eligible'] is False,label+': solo numerical benchmark scope; no automatic adoption')
    flags=report['backend_flags']
    check.require(not flags['tf32_matmul_allowed'] and not flags['tf32_cudnn_allowed'] and not flags['cudnn_benchmark'],label+': controlled TF32/cuDNN flags')
    if report['backend']=='math':
        check.require(flags['sdpa_math_enabled'] and not any(flags[key] for key in
            ('sdpa_flash_enabled','sdpa_mem_efficient_enabled','sdpa_cudnn_enabled')),label+': exclusive forced MATH backend')
    check.require(report['precision']==('fp32' if report['mode'].startswith('fp32') else 'bf16'),label+': mode precision')
    check.require(report['tdt_implementation']==('original' if report['mode'] in ('fp32-reference','bf16-reference') else 'v2'),label+': mode loss implementation')
    evidence_path=resolved(report['evidence_path'],check,label+': evidence')
    check.require(sha(evidence_path)==report['evidence_sha256']==entry['evidence_sha256'],label+': actual evidence SHA')
    check.require(evidence_path==Path(entry['evidence_path']).resolve(),label+': matrix evidence path')
    return report


def audit_profile(report, args, check, snapshot, pool, reports_by_evidence, data_cache):
    label=report['case_id']
    if report['status'] in ('error','error_oom'):
        return {'case_id':label,'family':report['family'],'batch_size':report['batch_size'],
                'precision':report['precision'],'backend':report['backend'],'mode':report['mode'],
                'status':report['status'],'error':report['error'],'timing':{},'adoption_eligible':False,
                'scope':'Retained failed invocation; no completed tensors, memory maximum or speed verified.'}
    identity=report['identity']
    check.require(identity['weights_sha256']==sha(args.weights),label+': provided frozen checkpoint bytes')
    check.require(identity['pool_sha256']==sha(args.pool),label+': provided frozen pool bytes')
    check.require(identity['source_sha256']==report['source_sha256'],label+': code identity')
    check.require(identity['base_revision']=='541d1f99c6b0c3cd0b11a95167540bb8edefd82b',label+': pinned v3 encoder/teacher revision')
    expected_ids=selected_ids(pool,report['family'])
    check.require(report['source_ids']==identity['source_ids']==expected_ids,label+': independently selected typical/long sources')
    family=report['family']
    if family not in data_cache:
        data_cache[family]=training_inputs(args.pool,expected_ids,check,label)
    rows,items=data_cache[family]
    check.require(identity['wav_sha256']==[item['wav_sha256'] for item in items],label+': exact ordered WAV bytes')
    check.require(identity['cache_sha256']==[item['cache_sha256'] for item in items],label+': exact ordered cache bytes')
    namespaces={Path(row['cache_path']).parent.name for row in rows}
    check.require(namespaces=={identity['cache_namespace']},label+': original teacher cache namespace')
    check.require(len(report['source_clips'])==16,label+': all source metadata')
    for spec,row,item in zip(report['source_clips'],rows,items):
        for key,value in row.items():
            check.require(spec[key]==value,label+': exact accepted pool '+key+' '+row['source_id'])
        check.require(spec['labels']==item['labels'].tolist() and spec['cache_sha256']==item['cache_sha256'],label+': source labels/cache digest')
        check.near(spec['actual_pcm_seconds'],row['duration'],label+': measured PCM seconds',atol=1e-5)
        check.require(spec['features_dtype']==str(item['features'].dtype),label+': cached feature storage dtype')
    audio=math.fsum(row['duration'] for row in rows)
    check.near(report['audio_seconds_per_update'],audio,label+': actual16 recording durations')
    evidence=torch.load(report['evidence_path'],map_location='cpu',weights_only=True)
    check.require(evidence['identity']==identity and evidence['backend_flags']==report['backend_flags'],label+': evidence/report identity and backend')
    check.require(evidence['reference_batch_size']==report['batch_size'],label+': evidence physical batch')
    check.require(set(evidence['gradients'])==set(snapshot['weights']) and len(evidence['gradients'])==report['gradient_tensors']==207,label+': actual full207 gradient layout')
    check.require(report['gradient_elements']==sum(value.numel() for value in snapshot['weights'].values()),label+': exact full trainable element count')
    check.require(report['initial_trainable_sha256']==state_digest(snapshot['weights']),label+': identical initial trainable state digest')
    masking=mask_audit(evidence['microbatch_tensors'],rows,items,report['batch_size'],check,label)
    check.require(report['groups']==[batch['indices'] for batch in evidence['microbatch_tensors']],label+': exact group manifest')
    check.require(evidence['microbatches']==report['microbatches'] and len(report['microbatches'])==report['accumulation'],label+': all microbatch metadata')
    for saved,batch in zip(report['microbatches'],evidence['microbatch_tensors']):
        mask=batch['input_attention_mask'].bool()
        for key,value in {'indices':batch['indices'],'batch_size':len(batch['indices']),
                          'input_features_shape':list(batch['input_features'].shape),'attention_mask_shape':list(mask.shape),
                          'input_mask_sums':mask.sum(-1).tolist(),'encoder_frames':batch['frame_lengths'].tolist(),
                          'target_tokens':batch['target_lengths'].tolist(),'labels_shape':list(batch['labels'].shape),
                          'decoder_input_ids_shape':list(batch['decoder_input_ids'].shape),'replay_indices':batch['replay_indices'],
                          'input_padding_frames':int(mask.numel()-mask.sum())}.items():
            check.require(saved[key]==value,label+': saved microbatch '+key)
        check.require(all(saved[key] is True for key in ('right_padded_mask','input_padding_zero','cache_padding_zero','target_padding_blank')),
                      label+': saved padding invariants agree with tensors')
        ts=batch['frame_lengths'].tolist();us=batch['target_lengths'].tolist()
        check.near(saved['lattice_padding_factor'],len(ts)*max(ts)*(max(us)+1)/sum(t*(u+1) for t,u in zip(ts,us)),label+': microbatch lattice padding')
    padding=report['padding']
    check.require(padding['valid_lattice_states']==masking['valid_lattice_cells'] and padding['padded_lattice_states']==masking['padded_lattice_cells'],label+': actual lattice totals')
    check.near(padding['lattice_padding_factor'],masking['lattice_padding_factor'],label+': aggregate lattice padding')
    expected_encoder=sum(len(batch['indices'])*int(batch['frame_lengths'].max())-int(batch['frame_lengths'].sum()) for batch in evidence['microbatch_tensors'])
    expected_decoder=sum(len(batch['indices'])*int(batch['target_lengths'].max())-int(batch['target_lengths'].sum()) for batch in evidence['microbatch_tensors'])
    check.require(padding['encoder_padding_frames']==expected_encoder and padding['decoder_padding_tokens']==expected_decoder,label+': exact encoder/decoder padding totals')
    per_clip_audit(evidence,rows,check,label)
    check.require(len(evidence['encoder_features'])==16,label+': all16 captured valid features')
    for index,value in enumerate(evidence['encoder_features']):
        tensor_finite(value,check,label+': valid student feature '+str(index))
        check.require(value.shape==(1,items[index]['frame_length'],1024),label+': valid student feature dimensions '+str(index))
    adam=adam_update_audit(evidence,snapshot,check,label)
    optimizer=report['optimizer'];update=evidence['adam_update']
    for key,expected in {'head_lr':update['lr'],'betas':update['betas'],'eps':update['eps'],
                         'head_weight_decay':update['weight_decay'],'lora_weight_decay':0.,'clip_norm':1.,
                         'new_token_lr_multiplier':15.,'old_size':8193,'extended_vocab_size':8199,
                         'optimizer_step_before_audit':0}.items():
        check.require(optimizer[key]==expected,label+': optimizer policy '+key)
    check.require(math.isfinite(optimizer['lora_lr']) and optimizer['lora_lr']>0,label+': positive adapter LR')
    comparison=None;reference_meta=None
    if report['reference_path']:
        refpath=resolved(report['reference_path'],check,label+': reference')
        check.require(sha(refpath)==report['reference_sha256'],label+': exact reference file SHA')
        check.require(str(refpath) in reports_by_evidence,label+': reference has an actual matrix profile')
        reference_meta=reports_by_evidence[str(refpath)]
        reference=torch.load(refpath,map_location='cpu',weights_only=True)
        same=report['mode']!='fp32-parity'
        if same:
            check.require(reference_meta['mode']=='bf16-reference' and reference_meta['tdt_implementation']=='original'
                          and reference_meta['batch_size']==report['batch_size']
                          and reference_meta['backend_flags']==report['backend_flags'],label+': identical-shape/backend original BF16 reference')
        else:
            check.require(reference_meta['mode']=='fp32-reference' and reference_meta['batch_size']==1
                          and reference_meta['backend']==report['backend']=='math',label+': native singleton forced-MATH FP32 oracle')
        comparison=correctness_comparison(reference,evidence,check,label,same_shape=same,include_weighted_bound=not same)
        compare_recorded(report['parity'],comparison,check,label)
        check.require(report['parity']['passed']==comparison['strict_gate']['passed']
                      and report['parity']['strict_gate']['passed']==comparison['strict_gate']['passed'],label+': unchanged numerical gate result independently reconstructed')
        del reference
    else:
        check.require(report['mode'] in ('fp32-reference','bf16-reference') and report['parity']['passed'] is None,
                      label+': reference-only capture without invented parity')
    failed=comparison is not None and not comparison['strict_gate']['passed']
    expected_status=('measured_with_numerical_gate_failure' if failed else 'measured') if report['mode']=='timing' else ('strict_parity_failed' if failed else 'reference_saved')
    check.require(report['status']==expected_status,label+': exact numerical/status policy')
    peak=report['numerical_gpu_peak']
    check.require(0<peak['allocated_gb']<=peak['reserved_gb']+1e-9,label+': numerical-stage actual memory')
    if 'max_gpu_gb' in report:
        check.require(peak['allocated_gb']<=report['max_gpu_gb'],label+': declared numerical allocation ceiling')
    pause=pause_audit(report['pause_provenance'],check,label)
    timings={};intervals=[]
    if report['mode']=='timing':
        check.require(bool(report['timing']) and set(report['timing']).issubset({'end_to_end','prefetched_cpu_inputs'}),label+': explicit actual timing category')
        check.require(not failed or report['exploratory_timing_authorized'] is True,label+': failed-gate timing explicitly exploratory')
        for category,value in report['timing'].items():
            timing=timing_audit(value,audio,check,label+': '+category,
                                minimum_updates=3 if family=='long' else 10,
                                minimum_warmup=1 if family=='long' else 2)
            check.near(value['audio_seconds_seen'],audio*value['updates'],label+': measured repeated audio count')
            check.require(value['reset_starting_trainable_sha256']==report['initial_trainable_sha256'] and
                          value['reset_optimizer_moments_and_step'] is True and value['reset_reuses_warmed_optimizer_buffers'] is True,
                          label+': reset checkpoint and all optimizer moments/step after warmup')
            check.require(value['solo_gpu_operator_declared'] is True,label+': timing solo declaration')
            start=datetime.fromisoformat(value['started_at_utc']);end=datetime.fromisoformat(value['ended_at_utc'])
            check.require(start.tzinfo is not None and end.tzinfo is not None and end>=start,label+': aware ordered timing timestamps')
            check.require((end-start).total_seconds()+.02>=timing['total_measured_seconds'],label+': timed durations fit recorded wall interval')
            if pause['available']:
                check.require(start>=datetime.fromisoformat(pause['paused_at_utc']),label+': timing after cooperative checkpoint pause')
            inventory_before=inventory_audit(value['process_inventory_before'],check,label+': inventory before')
            inventory_after=inventory_audit(value['process_inventory_after'],check,label+': inventory after')
            check.require(inventory_before['self_pid']==inventory_after['self_pid'],label+': same profiler PID across timings')
            includes={'H2D','encoder/decoder/joint/loss/backward','scalar logging synchronization','gradient clipping','AdamW','new-token row update scaling'}
            if category=='end_to_end':includes.update({'WAV reads','teacher-cache tensor reads','frontend','CPU collate'})
            check.require(set(value['includes'])==includes,label+': exact timing inclusions '+category)
            check.require(set(value['excludes'])=={'development/test inference','checkpoint/report serialization','warmup',
                                                  'CUDA synchronization before timer','initial model/teacher loading'},label+': exact excluded timing work')
            check.require(len(value['component_losses'])==value['updates'] and all(len(x)==4 and all(math.isfinite(float(v)) for v in x)
                          for x in value['component_losses']),label+': finite all measured objective components')
            check.require(timing['peak_allocated_vram_gb']<=report.get('max_gpu_gb',14.5),label+': measured allocation ceiling')
            timing.update({'started_at_utc':value['started_at_utc'],'ended_at_utc':value['ended_at_utc'],
                           'process_inventory_before':inventory_before,'process_inventory_after':inventory_after,
                           'scope':category,'includes':sorted(includes),'excludes':value['excludes']})
            timings[category]=timing;intervals.append((start,end,label+':'+category))
    else:
        check.require(not report['timing'],label+': correctness/reference capture has no speed claim')
    result={'case_id':label,'family':family,'batch_size':report['batch_size'],'accumulation':report['accumulation'],
            'effective_batch':16,'mode':report['mode'],'precision':report['precision'],'backend':report['backend'],
            'status':report['status'],'hardware':report['hardware'],'torch_version':report['torch_version'],
            'identity':identity,'optimizer':optimizer,'initial_trainable_sha256':report['initial_trainable_sha256'],
            'objective_coefficients':report['objective_coefficients'],'audio_seconds_per_update':audio,
            'masking':masking,'adam_update':adam,'numerical_comparison':comparison,'numerical_gpu_peak':peak,
            'timing':timings,'pause_provenance':pause,'adoption_eligible':False,
            'evidence_path':report['evidence_path'],'evidence_sha256':report['evidence_sha256']}
    del evidence
    return result,intervals


def comparison_table(profiles,check):
    timings=[row for row in profiles if row.get('timing')]
    table=[]
    for row in timings:
        for category,value in row['timing'].items():
            controls=[other for other in timings if other['batch_size']==1 and other['family']==row['family']
                      and other['precision']==row['precision'] and other['backend']==row['backend'] and category in other['timing']]
            check.require(len(controls)==1,row['case_id']+': unique matched B1 timing control '+category)
            baseline=controls[0]
            for key in ('identity','optimizer','hardware','torch_version','initial_trainable_sha256','objective_coefficients'):
                check.require(row[key]==baseline[key],row['case_id']+': matched timing '+key)
            control=baseline['timing'][category]
            check.require(value['updates']==control['updates'] and value['warmup_updates']==control['warmup_updates'],row['case_id']+': equal warmup/measured update counts')
            fp32=[other for other in profiles if other['family']==row['family'] and other['batch_size']==row['batch_size']
                  and other['mode']=='fp32-parity' and other['backend']=='math']
            fp32_gate=(fp32[0]['numerical_comparison']['strict_gate']['passed'] if len(fp32)==1 else None)
            if row['batch_size']==1 and not fp32:
                fp32_gate=None  # Native reference is an oracle, not original-v2 B1 validation.
            bf16_gate=row['numerical_comparison']['strict_gate']['passed'] if row['numerical_comparison'] else None
            ratio=value['audio_min_per_wall_min']/control['audio_min_per_wall_min']
            table.append({'family':row['family'],'batch_size':row['batch_size'],'accumulation':row['accumulation'],
                          'effective_batch':16,'precision':row['precision'],'backend':row['backend'],'timing_scope':category,
                          **{key:value[key] for key in ('updates','warmup_updates','audio_min_per_wall_min','clips_per_second',
                             'median_seconds_per_update','p90_seconds_per_update','peak_allocated_vram_gb','peak_reserved_vram_gb')},
                          'gain_vs_matched_b1':ratio,'throughput_increase_percent':(ratio-1)*100,
                          'lattice_padding_factor':row['masking']['lattice_padding_factor'],
                          'fp32_math_oracle_gate_passed':fp32_gate,'same_shape_bf16_gate_passed':bf16_gate,
                          'exploratory_despite_numerical_failure':row['status']=='measured_with_numerical_gate_failure',
                          'adoption_eligible':False,'case_id':row['case_id'],'control_case_id':baseline['case_id']})
    return sorted(table,key=lambda row:(row['family'],row['backend'],row['timing_scope'],row['batch_size']))


def coverage_from_table(table,profiles):
    coverage={};complete=True
    for family in ('typical','long'):
        subset=[row for row in table if row['family']==family]
        groups={}
        for row in subset:
            key=(row['precision'],row['backend'],row['timing_scope'])
            groups.setdefault(key,set()).add(row['batch_size'])
        group_coverage=[{'precision':key[0],'backend':key[1],'timing_scope':key[2],
                         'timed_batches':sorted(batches),'missing_timed_batches':sorted({1,2,4,8,16}-batches)}
                        for key,batches in sorted(groups.items())]
        available={row['batch_size'] for row in subset}
        coverage[family]={'timed_batches':sorted(available),'missing_timed_batches':sorted({1,2,4,8,16}-available),
                          'comparison_groups':group_coverage,
                          'retained_failed_cases':[row['case_id'] for row in profiles if row['family']==family and row['status'] in ('error','error_oom')]}
        complete &= bool(group_coverage) and all(not group['missing_timed_batches'] for group in group_coverage)
    return coverage,bool(complete)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Actual large_batch_matrix.json')
    parser.add_argument('--pool',type=Path,default=HERE/'runs/adaptive_20261010_01_screen_A/training_pool.json')
    parser.add_argument('--weights',type=Path,default=ROOT/'work/lora_20261010_01_best.pt')
    parser.add_argument('--output',type=Path,default=HERE/'large_batch_results_verification.json')
    parser.add_argument('--speed-report',type=Path,default=HERE/'large_batch_speed_report.json',help='Derived comparison table; numerical failures stay explicit')
    parser.add_argument('--allow-partial',action='store_true',help='Audit present rows while retaining missing/OOM cases without claiming a complete timing table')
    args=parser.parse_args();check=Checks()
    result={'schema_version':1,'passed':False,'gpu_used_by_auditor':False,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'auditor_sha256':sha(__file__),'scope':'CPU audit of saved matched effective16 numerical/timing evidence. No model or CUDA execution.',
            'limitations':['16 fixed TRAIN clips repeatedly presented; these are microbenchmarks, not full-data throughput or ASR quality.',
                           'Effective16 changes the optimizer batch versus the adaptive training loop effective4; ratios compare only this effective16 family.',
                           'Same-shape loss equivalence and FP32 MATH agreement do not establish acceptable BF16 batch-shape drift or authorize adoption.',
                           'Timing is parent/operator-declared solo GPU with saved Windows driver inventories; residual graphics contexts may remain.',
                           'Long-case timing may contain only3 measured updates after1 warmup; treat it as exploratory stress evidence.']}
    try:
        matrix=read(args.input)
        check.require(matrix['schema_version']==1 and matrix['effective_batch']==16 and bool(matrix['cases']),'actual effective16 matrix')
        check.require(len({row['case_id'] for row in matrix['cases']})==len(matrix['cases']),'unique actual matrix cases')
        check.require(len({row['report_path'] for row in matrix['cases']})==len(matrix['cases']),'unique actual report files')
        reports=[metadata_audit(row['report_path'],row,check) for row in matrix['cases']]
        evidence_map={str(Path(row['evidence_path']).resolve()):row for row in reports if row.get('evidence_path')}
        check.require(len(evidence_map)==sum(bool(row.get('evidence_path')) for row in reports),'unique actual evidence files')
        snapshot=torch.load(args.weights,map_location='cpu',weights_only=True)
        check.require(len(snapshot['weights'])==207 and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in snapshot['weights'].values()),'full207 finite FP32 starting checkpoint')
        pool=read(args.pool);data_cache={};profiles=[];intervals=[]
        for report in reports:
            audited=audit_profile(report,args,check,snapshot,pool,evidence_map,data_cache)
            if isinstance(audited,tuple):
                profile,times=audited;profiles.append(profile);intervals.extend(times)
            else:profiles.append(audited)
        for (start,end,label),(next_start,next_end,next_label) in zip(sorted(intervals),sorted(intervals)[1:]):
            check.require(end<=next_start,label+': no overlapping timed GPU jobs with '+next_label)
        table=comparison_table(profiles,check)
        coverage,complete=coverage_from_table(table,profiles)
        check.require(complete or args.allow_partial,'complete typical+long timing coverage, or explicitly allowed partial matrix')
        result.update({'passed':True,'status':'complete' if complete else 'verified_partial','matrix_path':str(args.input.resolve()),
                       'matrix_sha256_at_audit':sha(args.input),'checkpoint_sha256':sha(args.weights),'pool_sha256':sha(args.pool),
                       'profiles':profiles,'table':table,'coverage':coverage,'adoption_eligible':False,
                       'fastest_measured_by_family':{family:max((row for row in table if row['family']==family and row['timing_scope']=='end_to_end'),
                           key=lambda row:row['audio_min_per_wall_min'],default=None) for family in ('typical','long')}})
    except Exception as error:
        result.update({'status':'failed','error':str(error),'error_type':type(error).__name__})
    result['checks_count']=len(check.items);result['checks']=check.items
    write(args.output,result)
    speed={'schema_version':1,'status':result['status'],'integrity_passed':result['passed'],
           'verification_path':str(args.output.resolve()),'verification_sha256':sha(args.output),
           'auditor_sha256':sha(__file__),'effective_optimizer_batch':16,
           'table':result.get('table',[]),'coverage':result.get('coverage',{}),
           'fastest_measured_by_family':result.get('fastest_measured_by_family',{}),
           'adoption_eligible':False,'limitations':result['limitations'],
           'interpretation':'Ratios use matched B1x16 controls within identical16 TRAIN recordings/checkpoint/objective/precision/backend/timing scope. Numerical gate failures are preserved and exploratory timing is not permission to adopt a batch.'}
    write(args.speed_report,speed)
    print(json.dumps({'passed':result['passed'],'status':result['status'],'checks':len(check.items),'output':str(args.output),'error':result.get('error')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

