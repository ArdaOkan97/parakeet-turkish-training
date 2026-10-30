"""Matched TRAIN-only B8x2 custom-v2/native-NeMo training pilot.

The CPU preparation command never loads a model. GPU arms are explicit, separate
root-owned invocations under the successful isolated CUDA12.9 linker preload.
Compilation/cold call, warmup and weight/Adam/RNG reset precede timed updates.
Nothing is exported, evaluated on heldout data, or adopted by this profiler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE_RUN = 'adaptive_20261010_continuation_03_resume_B8'
DEFAULT_POOL = HERE / 'runs' / SOURCE_RUN / 'training_pool.json'
DEFAULT_CONFIG = HERE / 'runs' / SOURCE_RUN / 'run_config.json'
DEFAULT_WEIGHTS = ROOT / 'work' / (SOURCE_RUN + '_best.pt')
WEIGHTS_SHA = '4790b0a53c0e99ba6ea1a2b8bd9bbdd1d9787ca80b9a38ea6107701c098b98ed'
ADAPTER_SHA = 'f41d19bc88b82e6a03b394e8e42d48820314aa7a3a3d189b959c5503c2e57a7c'
PRELOAD_SHA = 'f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba'
NAMES = ('profile_native_training_v1.py', 'profile_batched_training.py',
         'profile_large_batches.py', 'batched_lora.py', 'efficient_tdt.py',
         'efficient_tdt_v2.py', 'native_tdt_checkpointed_v1.py',
         'preload_nemo_linker_v1.py', 'launch_nemo_linker_v4.py', 'runtime.py',
         'encoder_lora.py', 'extend_vocab.py', 'train_lora.py', 'train.py',
         'train_lora_batched_data_v4.py', 'public_turkish_data_policy_v3.py',
         'resource_watchdog_v2.py', 'paired_windows_commit_telemetry_v1.py',
         'diagnose_batch_objective_math.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                    sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sources():
    return {name: sha(HERE / name) for name in NAMES}


def utc():
    return datetime.now(timezone.utc).isoformat()


def totals(rows):
    tr = math.fsum(float(r['duration']) for r in rows if r['language'] == 'tr')
    replay = math.fsum(float(r['duration']) for r in rows if r['language'] != 'tr')
    return {'turkish_audio_seconds': tr, 'replay_audio_seconds': replay,
            'total_audio_seconds': tr + replay}


def summarize_updates(updates):
    require(bool(updates), 'At least one finite measured update is required')
    wall = math.fsum(r['wall_seconds'] for r in updates)
    require(wall > 0 and all(math.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0 for r in updates),
            'Finite positive synchronized update times are required')
    tr = math.fsum(r['turkish_audio_seconds'] for r in updates)
    replay = math.fsum(r['replay_audio_seconds'] for r in updates)
    ordered = sorted(r['wall_seconds'] for r in updates)
    return {'completed_updates': len(updates), 'measured_wall_seconds': wall,
            'turkish_audio_seconds': tr, 'replay_audio_seconds': replay, 'total_audio_seconds': tr + replay,
            'combined_audio_minutes_per_wall_minute': (tr + replay) / wall,
            'turkish_audio_minutes_per_wall_minute': tr / wall,
            'updates_per_second': len(updates) / wall,
            'median_seconds_per_update': statistics.median(ordered),
            'p90_seconds_per_update': ordered[min(len(ordered) - 1, int(.9 * len(ordered)))]}


def validate_plan(plan):
    require(plan['kind'] == 'matched_native_training_plan_v1' and plan['status'] == 'CPU_prepared', 'Wrong/incomplete plan')
    require(plan['runtime']['batch_size'] == 8 and plan['runtime']['accumulation'] == 2, 'Only matched B8x2 is supported')
    require(plan['weights']['sha256'] == WEIGHTS_SHA and plan['weights']['selected_step'] == 1871, 'Source1871 changed')
    require(len(plan['updates']) >= 1, 'Empty TRAIN update plan')
    for expected, update in enumerate(plan['updates'], 1):
        require(update['update'] == expected and len(update['batches']) == 2, 'Wrong update order/accumulation')
        rows = [row for group in update['batches'] for row in group]
        require(len(rows) == 16 and len({r['source_id'] for r in rows}) == 16, 'Each update requires sixteen distinct clips')
        require(all(len(group) == 8 for group in update['batches']), 'Physical batch must be eight')
        require(all(r['dataset_split'] == 'train' for r in rows), 'Heldout sources are forbidden')
        require(sum(r['language'] == 'tr' for r in rows) == 8, 'Exactly eight TR/eight replay clips required')
        for row in rows:
            t, u = row['encoder_frames'], row['target_tokens']
            require(type(t) is int and type(u) is int and t > 0 and 0 < u <= 180, 'Invalid actual accepted T/U')
            require(t * (u + 1) * 8204 * 4 <= 1_600_000_000, 'Native full-lattice safety bound exceeded; no crop allowed')
            require(math.isfinite(row['duration']) and row['duration'] > 0, 'Invalid audio duration')
        require(update['source_ids'] == [r['source_id'] for r in rows], 'Ordered TRAIN identity differs')
        require(update['audio'] == totals(rows), 'Audio numerator differs')
    require(plan['sampled_source_sequence_sha256'] == json_sha([u['source_ids'] for u in plan['updates']]), 'Source sequence differs')


def prepare(args):
    require(not args.plan.exists(), 'Refusing to overwrite an existing immutable plan')
    require(sha(DEFAULT_WEIGHTS) == WEIGHTS_SHA, 'Reviewed source1871 weights changed')
    require(sha(HERE / 'native_tdt_checkpointed_v1.py') == ADAPTER_SHA, 'Native adapter changed')
    require(sha(HERE / 'preload_nemo_linker_v1.py') == PRELOAD_SHA, 'Coherent preload changed')
    from train_lora_batched_data_v4 import BalancedBucketSampler
    import torch
    config, pool = read(DEFAULT_CONFIG), read(DEFAULT_POOL)
    expected = {'batch_size':8, 'accumulation':2, 'rank':8, 'alpha':16.0,
                'head_lr':5e-5, 'lora_lr':3e-4, 'new_token_lr_multiplier':6.0,
                'replay_fraction':.5, 'replay_kl':5.0, 'duration_kl':.5,
                'feature_distillation':1.0, 'warmup_steps':25, 'schedule':'constant',
                'chunk_size':16, 'bucket_size':128, 'seed':20261010, 'no_encoder_checkpoint':False}
    require(all(config.get(k) == v for k, v in expected.items()), 'Reviewed source runtime differs')
    sampler = BalancedBucketSampler(pool['turkish'], pool['replay'], seed=config['seed'],
                                    batch_size=8, accumulation=2, replay_fraction=.5, bucket_size=128)
    updates = []
    for index in range(1, args.updates + 1):
        groups = sampler.next_batches()
        rows = [r for group in groups for r in group]
        updates.append({'update':index, 'batches':groups, 'source_ids':[r['source_id'] for r in rows], 'audio':totals(rows)})
    plan = {'schema_version':1, 'kind':'matched_native_training_plan_v1', 'status':'CPU_prepared',
            'created_at_utc':utc(), 'source_run_id':SOURCE_RUN,
            'weights':{'path':str(DEFAULT_WEIGHTS), 'sha256':WEIGHTS_SHA, 'selected_step':1871},
            'pool':{'path':str(DEFAULT_POOL), 'sha256':sha(DEFAULT_POOL),
                    'turkish_clips':len(pool['turkish']), 'replay_clips':len(pool['replay']),
                    'scope':'Original accepted/cached TRAIN pool; current expanded pool preparation is separate'},
            'config':{'path':str(DEFAULT_CONFIG), 'sha256':sha(DEFAULT_CONFIG)},
            'runtime':expected, 'updates':updates,
            'sampled_source_sequence_sha256':json_sha([u['source_ids'] for u in updates]),
            'code_sha256':sources(), 'cuda_initialized':torch.cuda.is_initialized(),
            'notes':['Fresh Adam and sampler draw order for a changed loss speed pilot; not exact resume of source1871.',
                     'Neither heldout predictions nor inference are used.',
                     'Actual selected WAV/cache bytes, PCM lengths and token IDs are verified before GPU timing.',
                     'Native sigma0/omega0/raw-none keeps current per-target clip normalization; no released-recipe equivalence claim.']}
    validate_plan(plan)
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA')
    write(args.plan, plan)
    print(json.dumps({'status':plan['status'], 'plan':str(args.plan), 'sha256':sha(args.plan), 'updates':len(updates),
                      'unique_train_sources':len({s for u in updates for s in u['source_ids']}), 'cuda_initialized':False}), flush=True)


def compare(args):
    require(not args.output.exists(), 'Comparison output already exists')
    a, b = read(args.custom_report), read(args.native_report)
    require(a['arm'] == 'custom' and b['arm'] == 'native', 'Expected custom/native report ordering')
    require(a['identity'] == b['identity'], 'Matched starting weights/inputs/runtime/code/environment differs')
    require(a.get('reset') == b.get('reset') and a.get('reset',{}).get('optimizer_state_empty') is True,
            'Fresh Adam/starting trainables/Python and CPU-CUDA RNG reset evidence differs')
    require(a.get('frozen_sources_unchanged') is True and b.get('frozen_sources_unchanged') is True and
            a.get('weights_sha256_after') == WEIGHTS_SHA and b.get('weights_sha256_after') == WEIGHTS_SHA,
            'Frozen runtime or original source weights changed')
    common = min(len(a['updates']), len(b['updates']))
    require(common > 0, 'No common measured optimizer prefix')
    for x, y in zip(a['updates'][:common], b['updates'][:common]):
        require(x['update'] == y['update'] and x['source_ids'] == y['source_ids'] and x['audio'] == y['audio'],
                'Actual matched TRAIN/audio order differs')
    left, right = summarize_updates(a['updates'][:common]), summarize_updates(b['updates'][:common])
    complete = (a['status'] == b['status'] == 'complete' and
                len(a['updates']) == len(b['updates']) == a['requested_updates'] == b['requested_updates'])
    cold_deltas = [abs(x-y) for x,y in zip(a['cold_start']['update']['components'],b['cold_start']['update']['components'])]
    value = {'schema_version':1, 'kind':'matched_native_training_comparison_v1',
             'status':'complete' if complete else 'partial_matched_prefix',
             'created_at_utc':utc(), 'common_measured_updates':common,
             'arms':{name:{'status':r['status'],'requested_updates':r['requested_updates'],
                           'completed_updates':len(r['updates']),'failure_reason':r.get('error')}
                     for name,r in (('custom',a),('native',b))},
             'custom_report':{'path':str(args.custom_report),'sha256':sha(args.custom_report)},
             'native_report':{'path':str(args.native_report),'sha256':sha(args.native_report)},
             'identity':a['identity'], 'custom':left, 'native':right,
             'native_over_custom_training_speed_ratio':left['measured_wall_seconds']/right['measured_wall_seconds'],
             'cold_start':{'custom':a['cold_start'],'native':b['cold_start']},
             'memory':{'custom':a['memory'],'native':b['memory']},
             'numerical_comparison':{'first_objective_components_measured':True, 'component_absolute_deltas':cold_deltas,
                                    'full_gradients_measured':False, 'strict_gradient_gate_passed':None,
                                    'reason':'Full gradient capture optional; scalar agreement alone does not prove gradient equivalence.'},
             'adoption_authorized':False, 'heldout_used':False}
    if a.get('evidence_path') and b.get('evidence_path'):
        import torch
        from diagnose_batch_objective_math import compare_tensors, strict_gate
        require(sha(a['evidence_path']) == a['evidence_sha256'] and sha(b['evidence_path']) == b['evidence_sha256'], 'Gradient evidence changed')
        evidence_a = torch.load(a['evidence_path'], map_location='cpu', weights_only=True)
        evidence_b = torch.load(b['evidence_path'], map_location='cpu', weights_only=True)
        result = compare_tensors(evidence_a, evidence_b)
        value['numerical_comparison'] = {'full_gradients_measured':True, 'comparison':result, 'strict_same_shape_BF16_gate':strict_gate(result, same_shape=True),
                                         'scope':'Same-shape first B8x2 objective gradients; a failure remains descriptive, never waived'}
    write(args.output, value)
    print(json.dumps({'comparison':str(args.output), 'common_updates':common,
                      'training_speed_ratio':value['native_over_custom_training_speed_ratio']}), flush=True)


def run(args):
    require(args.gpu_window_authorized, 'Root-owned exclusive GPU window flag is required')
    require(not args.output.exists(), 'Refusing to overwrite measured output')
    evidence_path = args.output.with_suffix('.first_update.pt')
    require(not evidence_path.exists(), 'Refusing to overwrite correctness evidence')
    plan = read(args.plan)
    validate_plan(plan)
    code = sources()
    require(code == plan['code_sha256'], 'Prepared source fingerprint differs; use a reviewed new plan')
    require(sha(plan['weights']['path']) == plan['weights']['sha256'] and
            sha(plan['pool']['path']) == plan['pool']['sha256'] and sha(plan['config']['path']) == plan['config']['sha256'],
            'Reviewed weights/pool/config changed')
    require(args.updates <= len(plan['updates']), 'Requested updates exceed the prepared sequence')
    require(sha(HERE / 'preload_nemo_linker_v1.py') == PRELOAD_SHA and
            sha(HERE / 'native_tdt_checkpointed_v1.py') == ADAPTER_SHA, 'Native/preload dependency changed')
    from preload_nemo_linker_v1 import preload
    preload_started = time.perf_counter()
    preloaded = preload()  # Must precede every Torch/Numba/adapter import.
    preload_wall_seconds = time.perf_counter() - preload_started
    import torch
    import soundfile as sf
    from profile_batched_training import initialize
    from profile_large_batches import state_digest, batch_invariants, backend_flags, gpu_process_evidence
    from batched_lora import collate_training_batch, batch_objective
    from train_lora_batched_data_v4 import lr_factor
    from resource_watchdog_v2 import ResourceGuard, sample_memory, worker_process_commit_reason
    from native_tdt_checkpointed_v1 import configuration_identity
    from efficient_tdt_v2 import chunked_tdt_loss as custom_loss
    if args.arm == 'native':
        from native_tdt_checkpointed_v1 import chunked_tdt_loss as loss_function
    else:
        loss_function = custom_loss
    torch.set_num_threads(args.cpu_threads)
    total = torch.cuda.get_device_properties(0).total_memory
    require(total >= 9_000_000_000, 'Physical GPU too small for the predeclared 9GB allocation ceiling')
    torch.cuda.set_per_process_memory_fraction(9_000_000_000 / total)
    # Same fresh-process defaults as the current training job. Record actual flags;
    # do not claim historical strict BF16 cross-shape equivalence.
    backend = backend_flags()
    process_start = time.perf_counter()
    report = {'schema_version':1, 'kind':'matched_native_training_profile_v1', 'arm':args.arm,
              'status':'running', 'created_at_utc':utc(), 'updates':[], 'cold_start':{}, 'warmup':[],
              'requested_updates':args.updates, 'requested_warmups':args.warmup,
              'gpu_window_context':args.gpu_context, 'pid':os.getpid(), 'ppid':os.getppid(),
              'preload_wall_seconds_excluded_from_training_rate':preload_wall_seconds,
              'preload':preloaded, 'backend_flags':backend, 'native_loss_configuration':configuration_identity(),
              'numerical_gate_required_for_adoption':True, 'adoption_authorized':False, 'heldout_used':False,
              'memory':{}, 'timing_scope':'WAV/cache reads, frontend, H2D, full forward/loss/backward, clipping, fresh AdamW and new-row scaling, sync and optimizer-end cache release; telemetry/publication outside update time'}
    guard = ResourceGuard()
    observations, selected = [], {}
    def observe(phase):
        sample = sample_memory(os.getpid(), phase)
        reason, _ = guard.observe(sample['system_memory'], sample['worker_private_memory'], observed_monotonic=sample['observed_monotonic'])
        reason = reason or worker_process_commit_reason(sample['calling_process_commit_memory'], os.getpid())
        observations.append(sample)
        require(reason is None, 'Resource policy stop: ' + str(reason))
        require(time.perf_counter() - process_start < args.max_wall_seconds, 'Bounded profile wall deadline reached')
        return sample
    try:
        observe('pre_model_setup')
        # Verify each actual selected asset once; no whole-dataset rescans.
        for update in plan['updates'][:args.updates]:
            for group in update['batches']:
                for row in group:
                    if row['source_id'] in selected:
                        continue
                    info = sf.info(row['audio_filepath'])
                    require(info.samplerate == 16000 and info.channels == 1 and abs(info.frames/16000-row['duration']) < 1e-8, 'Actual WAV PCM duration/format changed')
                    require(sha(row['audio_filepath']) == row['wav_sha256'], 'Selected TRAIN WAV changed')
                    cache = torch.load(row['cache_path'], map_location='cpu', weights_only=True)
                    require(cache['source_id'] == row['source_id'] and cache['features'].shape == (row['encoder_frames'],1024)
                            and cache['labels'].shape == (row['target_tokens'],) and torch.isfinite(cache['features']).all(), 'Selected TRAIN cache changed')
                    labels = cache['labels'].long()
                    require(bool(((labels>=0)&(labels<8199)&(labels!=8192)).all()), 'Invalid lexical target in actual cache')
                    require(row['language']=='tr' or int(labels.max())<8193, 'Replay target cannot enter old teacher embedding')
                    selected[row['source_id']] = {'source_id':row['source_id'], 'wav_sha256':row['wav_sha256'],
                                                'cache_sha256':sha(row['cache_path']), 'actual_pcm_seconds':info.frames/16000,
                                                'encoder_frames':row['encoder_frames'], 'target_tokens':row['target_tokens'],
                                                'labels_sha256':hashlib.sha256(labels.numpy().tobytes()).hexdigest()}
                    del cache, labels
        identity = {'source1871_weights_sha256':plan['weights']['sha256'], 'plan_sha256':sha(args.plan),
                    'pool_sha256':plan['pool']['sha256'], 'source_config_sha256':plan['config']['sha256'],
                    'runtime':plan['runtime'], 'precision':'bf16_autocast_FP32_trainables', 'code_sha256':code,
                    'source_ids':plan['updates'][0]['source_ids'], 'assets':sorted(selected.values(), key=lambda r:r['source_id']),
                    'measured_order_sha256':json_sha([u['source_ids'] for u in plan['updates'][:args.updates]]),
                    'backend_flags':backend, 'torch_version':torch.__version__, 'cuda_compiler_binding':preloaded['actual_binding_linker']}
        report['identity'] = identity
        report['solo_gpu_process_inventory_before'] = gpu_process_evidence()
        model, processor, teacher, old_size, checkpoint = initialize(SimpleNamespace(weights=plan['weights']['path'], no_encoder_checkpoint=False))
        require(checkpoint['step']==1871 and len(checkpoint['weights'])==207, 'Wrong checkpoint step/layout')
        parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        require(len(parameters)==207, 'Unexpected trainable layout')
        initial = {n:w.detach().cpu().clone() for n,w in checkpoint['weights'].items()}
        initial_digest = state_digest(initial)
        report['initial_trainables_sha256'] = initial_digest
        del checkpoint
        heads = [p for n,p in parameters if not n.endswith(('lora_A','lora_B'))]
        adapters = [p for n,p in parameters if n.endswith(('lora_A','lora_B'))]
        row_parameters = [model.decoder.embedding.weight, model.joint.head.weight, model.joint.head.bias]
        optimizer = None
        rng = None
        def reset():
            nonlocal optimizer, rng
            model.load_state_dict(initial, strict=False)
            optimizer = torch.optim.AdamW([{'params':heads,'lr':5e-5,'weight_decay':.01},
                                           {'params':adapters,'lr':3e-4,'weight_decay':0.}], betas=(.9,.999), eps=1e-8)
            optimizer.zero_grad(set_to_none=True)
            random.seed(20261010); torch.manual_seed(20261010); torch.cuda.manual_seed_all(20261010)
            rng = {'python':random.getstate(), 'torch_cpu':torch.get_rng_state(), 'torch_cuda':torch.cuda.get_rng_state_all()}
            require(not optimizer.state and state_digest({n:p for n,p in parameters})==initial_digest, 'Weight/fresh Adam reset failed')
            torch.cuda.empty_cache()
        reset()
        def update(specification, step, capture=False):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            factor = lr_factor(step,args.updates,25,'constant')
            optimizer.param_groups[0]['lr']=5e-5*factor; optimizer.param_groups[1]['lr']=3e-4*factor
            components = [0.,0.,0.,0.]; per_clip={key:[] for key in ('tdt','token_kl','duration_kl','feature_distillation','total')}
            features, masks = [], []
            for group in specification['batches']:
                batch=collate_training_batch(processor,group,blank_id=model.config.blank_token_id,old_size=old_size,device='cuda')
                result=batch_objective(model,teacher,batch,old_size=old_size,replay_kl=5,duration_kl=.5,feature_distillation=1,
                                       chunk_size=16,tdt_loss_fn=loss_function,tdt_host_lengths=True,use_autocast=True,
                                       return_debug_features=capture)
                scaled=result.loss/2
                require(bool(torch.isfinite(scaled)), 'Nonfinite actual TRAIN objective')
                scaled.backward()
                metrics=result.detached_metrics()
                components=[a+metrics[k]/2 for a,k in zip(components,('tdt_loss','replay_kl','replay_duration_kl','replay_feature_distillation'))]
                if capture:
                    for key in per_clip: per_clip[key]+=result.per_clip[key].detach().float().cpu().tolist()
                    for i,t in enumerate(result.encoder_frames): features.append(result.debug_encoder_features[i:i+1,:t].cpu())
                    masks.append({'source_ids':[r['source_id'] for r in group], 'input_mask':batch.inputs['attention_mask'].cpu(),
                                  'labels':batch.labels.cpu(), 'decoder_input_ids':batch.decoder_input_ids.cpu(),
                                  'frame_lengths':batch.frame_lengths,'target_lengths':batch.target_lengths,
                                  'replay_indices':list(batch.replay_indices)})
                del result,scaled,batch
            evidence=None
            if capture:
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _,p in parameters), 'Missing/nonfinite trainable gradient')
                evidence={'identity':identity,'components':components,'per_clip':per_clip,'encoder_features':features,
                          'gradients':{n:p.grad.detach().cpu().clone() for n,p in parameters},'batches':masks}
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True)
            previous=[p[old_size:model.config.vocab_size].detach().clone() for p in row_parameters]
            optimizer.step()
            with torch.no_grad():
                for p,old in zip(row_parameters,previous):
                    new=p[old_size:model.config.vocab_size]; new.copy_(old+6*(new-old))
            torch.cuda.synchronize()
            reserved_before=torch.cuda.memory_reserved()
            cleanup_started=time.perf_counter(); cleaned=reserved_before>4_000_000_000
            if cleaned: torch.cuda.empty_cache()
            cleanup_seconds=time.perf_counter()-cleanup_started
            wall=time.perf_counter()-started
            record={'update':step,'source_ids':specification['source_ids'],'audio':specification['audio'], **specification['audio'],
                    'wall_seconds':wall,'components':components,'weighted_loss':sum(a*c for a,c in zip(components,(1,5,.5,1))),
                    'gradient_norm':float(norm),'head_lr':optimizer.param_groups[0]['lr'],'lora_lr':optimizer.param_groups[1]['lr'],
                    'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                    'reserved_before_cleanup_bytes':reserved_before,'cleanup_performed':cleaned,'cleanup_seconds':cleanup_seconds}
            return record,evidence
        # First complete call includes NeMo imports/JIT and checkpoint backward.
        cold_started=time.perf_counter()
        cold,evidence=update(plan['updates'][0],1,capture=args.capture_gradients)
        report['cold_start']={'wall_seconds_including_capture_and_JIT':time.perf_counter()-cold_started,'update':cold,
                              'excluded_from_training_rate':True,'full_native_checkpoint_backward_executed':args.arm=='native'}
        if evidence is not None:
            torch.save(evidence,evidence_path)
            report.update(evidence_path=str(evidence_path),evidence_sha256=sha(evidence_path))
            del evidence
        observe('after_first_JIT_and_backward')
        for i in range(args.warmup):
            record,_=update(plan['updates'][i % len(plan['updates'])],i+1)
            report['warmup'].append(record)
            observe('warmup')
        reset()
        report['reset']={'weights_sha256':initial_digest,'optimizer_state_empty':not bool(optimizer.state),
                         'seed':20261010,'python_rng_sha256':json_sha(rng['python']),
                         'torch_cpu_rng_sha256':hashlib.sha256(rng['torch_cpu'].numpy().tobytes()).hexdigest(),
                         'torch_cuda_rng_sha256':[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in rng['torch_cuda']],
                         'warmup_updates_discarded':True,'allocator_cap_bytes':9_000_000_000,'cleanup_threshold_bytes':4_000_000_000}
        torch.cuda.reset_peak_memory_stats()
        report['timed_start_at_utc']=utc()
        for specification in plan['updates'][:args.updates]:
            observe('before_timed_TRAIN_update')
            record,_=update(specification,specification['update'])
            report['updates'].append(record)
            observe('after_timed_TRAIN_update')
            report['timing']=summarize_updates(report['updates'])
            write(args.output,report)
            print(json.dumps({'arm':args.arm,'update':record['update'], 'loss':record['weighted_loss'],
                              'wall_seconds':record['wall_seconds'],'rates':report['timing']}),flush=True)
        report['status']='complete'
        report['final_trainables_sha256']=state_digest({n:p for n,p in parameters})
    except Exception as error:
        report['status']='partial_resource_or_failure' if report['updates'] else 'failed_before_measured_updates'
        report['error']=str(error); report['traceback']=traceback.format_exc(limit=12)
    finally:
        report['finished_at_utc']=utc(); report['setup_and_total_process_wall_seconds']=time.perf_counter()-process_start
        if report['updates']: report['timing']=summarize_updates(report['updates'])
        if torch.cuda.is_initialized():
            report['memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                              'final_allocated_bytes':torch.cuda.memory_allocated(),'final_reserved_bytes':torch.cuda.memory_reserved(),
                              'allocator_cap_bytes':9_000_000_000,'cleanup_threshold_bytes':4_000_000_000,
                              'system_max_commit_fraction':max((s['system_memory'].get('commit_fraction',0.) for s in observations),default=0.),
                              'actual_worker_max_private_bytes':max((s['worker_private_memory'].get('private_bytes',0) for s in observations),default=0)}
        report['memory_observations']=observations
        report['code_sha256_after']=sources()
        report['frozen_sources_unchanged']=report['code_sha256_after']==code
        report['weights_sha256_after']=sha(plan['weights']['path'])
        require(report['frozen_sources_unchanged'] and report['weights_sha256_after']==WEIGHTS_SHA, 'Profiler changed/observed changed sources or source weights')
        write(args.output,report)
    print(json.dumps({'status':report['status'],'output':str(args.output),'completed_updates':len(report['updates'])}),flush=True)
    return 0 if report['status']=='complete' else 1


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--compare',action='store_true')
    p.add_argument('--arm',choices=('custom','native'))
    p.add_argument('--plan',type=Path,default=HERE/'native_training_pilot_plan_v1.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--custom-report',type=Path)
    p.add_argument('--native-report',type=Path)
    p.add_argument('--updates',type=int,default=20)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--max-wall-seconds',type=float,default=600.)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--capture-gradients',action='store_true',help='Optional first-prefix full207 gradients/features; excluded from rates')
    p.add_argument('--gpu-window-authorized',action='store_true')
    p.add_argument('--gpu-context',default='root_owned_exclusive_training_benchmark_window')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    require(1<=args.updates<=100 and 2<=args.warmup<=5 and args.cpu_threads>0 and
            math.isfinite(args.max_wall_seconds) and 0<args.max_wall_seconds<=1800, 'Invalid bounded profile settings')
    require(sum((args.prepare,args.compare,bool(args.arm)))==1,'Select exactly prepare, compare or one GPU arm')
    if args.prepare: prepare(args)
    elif args.compare:
        require(args.output and args.custom_report and args.native_report,'Comparison requires both reports and new output')
        compare(args)
    else:
        require(args.output is not None,'A new per-arm output path is required')
        sys.exit(run(args))
