"""Bounded solo-GPU TRAIN-only variable-shape allocator A/B experiment.

Both branches restore the same full trained weights/Adam/sampler/RNG checkpoint.
Control uses the preceding13GB-cap/6GB-cleanup policy; intervention uses9GB/4GB.
Both branches collect actual self-PID host telemetry and enforce the new commit guard. This explicit harness policy is distinct from the frozen production hook.
CPU --prepare validates inputs and next source draws without initializing CUDA.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import torch
import soundfile as sf
from atomic_io_retry_v1 import read_bytes_shared, write_json
from batched_lora import collate_training_batch, batch_objective
from cuda_cache_reclaim_v1 import FROZEN_TRAINER_SHA256
from efficient_tdt_v2 import chunked_tdt_loss, clear_dp_plan_cache
from profile_batched_training import initialize
from runtime import ROOT, OUTPUT, DATA
from train_lora_batched import BalancedBucketSampler, ids_sha
from resource_watchdog_v1 import ResourceGuard, sample_memory, POLICY as RESOURCE_POLICY
from launch_with_memory_guard_v3 import creation_identity, validate_capacity


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''):result.update(chunk)
    return result.hexdigest()


def tensor_sha(value):
    value=value.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def tree_sha(value):
    digest=hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((str(item.dtype),tuple(item.shape))).encode());digest.update(bytes.fromhex(tensor_sha(item)))
        elif isinstance(item,dict):
            for key in sorted(item,key=lambda x:str(x)):digest.update(repr(key).encode());visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:visit(child)
        else:digest.update(repr(item).encode())
    visit(value);return digest.hexdigest()


def memory_sample(phase):
    value=sample_memory(os.getpid(),phase)
    system=value['system_memory']
    if system.get('available'):
        system['committed_bytes']=system['commit_used_bytes']  # legacy report alias
    return value


def system_memory():
    return memory_sample('probe_auxiliary_read')['system_memory']


def driver_memory():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        return {'returncode':result.returncode,'raw':result.stdout.strip(),'stderr':result.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as error:return {'error':str(error)}


def allocator_stats():
    stats=torch.cuda.memory_stats()
    keys=('allocated_bytes.all.current','reserved_bytes.all.current','active_bytes.all.current',
          'inactive_split_bytes.all.current','inactive_split.all.current','segment.all.current',
          'num_alloc_retries','num_ooms')
    return {key:int(stats.get(key,0)) for key in keys}


def safety_reason(memory,reserved_bytes,*,resource_guard=None,observed_monotonic=None,
                  worker_private_memory=None,reserved_limit=12_000_000_000):
    guard=ResourceGuard() if resource_guard is None else resource_guard
    reason,_=guard.observe(memory,worker_private_memory,observed_monotonic=observed_monotonic)
    if reason:return reason
    if reserved_bytes>=reserved_limit:return 'allocator_reservation_guard'
    return None


def summarize_snapshot(snapshot):
    totals=Counter();sites=Counter();inactive_segments=[]
    for segment in snapshot.get('segments',[]):
        totals['segments']+=1;totals['reserved_bytes']+=segment['total_size']
        whole=not segment.get('active_size',0)
        if whole:
            totals['fully_inactive_segment_bytes']+=segment['total_size']
            inactive_segments.append(segment['total_size'])
        for block in segment['blocks']:
            totals[block['state']+'_bytes']+=block['size']
            if block['state']=='inactive':
                if not whole:totals['inactive_in_live_segment_bytes']+=block['size']
                frames=block.get('frames',[])
                local=[f for f in frames if str(ROOT).replace('\\','/').lower() in f.get('filename','').replace('\\','/').lower()]
                frame=(local or frames or [{}])[0]
                site='{}:{}:{}'.format(frame.get('filename','unknown'),frame.get('line','?'),frame.get('name','unknown'))
                sites[site]+=block['size']
    traces=Counter(entry.get('action','unknown') for device in snapshot.get('device_traces',[]) for entry in device)
    return {'totals':dict(totals),'largest_fully_inactive_segments_bytes':sorted(inactive_segments,reverse=True)[:10],
            'largest_inactive_block_sites':[{'site':site,'bytes':size} for site,size in sites.most_common(12)],
            'bounded_trace_action_counts':dict(traces)}


def read_inputs(args):
    for path in (args.checkpoint,args.run_config,args.pool):
        if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():raise ValueError('Owned workspace input required')
    config=json.loads(read_bytes_shared(args.run_config));pool=json.loads(read_bytes_shared(args.pool))
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (state['run_id']!=config['run_id'] or state['config_identity']!=config['config_identity']
            or len(state['optimizer']['state'])!=207 or len(state['weights'])!=207):raise ValueError('Exact full checkpoint/config identity required')
    if (config['batch_size'],config['accumulation'],config['tdt_implementation'],config['chunk_size'])!=(8,2,'v2',16):
        raise ValueError('Probe is scoped to actual B8x2/v2/chunk16 configuration')
    if ids_sha(pool['turkish'])!=config['config_identity']['tr_pool_sha256'] or ids_sha(pool['replay'])!=config['config_identity']['replay_pool_sha256']:
        raise ValueError('Exact accepted source pool required')
    for name,digest in config['code_sha256'].items():
        if sha(OUTPUT/name)!=digest:raise ValueError('Frozen source differs: '+name)
    if sha(OUTPUT/'train_lora_batched.py')!=FROZEN_TRAINER_SHA256:raise ValueError('Frozen trainer mismatch')
    heldout_sources=set();heldout_audio=set();heldout_sentences=set()
    for language in ('en','es','it','fr','de','tr'):
        for split in ('validation','test'):
            path=DATA/f'{split}_{language}.jsonl'
            if sha(path)!=config['input_sha256'][str(path.resolve())]:raise ValueError('Protected manifest differs')
            for line in read_bytes_shared(path).decode().splitlines():
                row=json.loads(line);heldout_sources.add(row['source_id']);heldout_audio.add(row['wav_sha256'])
                if row.get('sentence_id') is not None:heldout_sentences.add(str(row['sentence_id']))
    for row in (*pool['turkish'],*pool['replay']):
        if (row['dataset_split']!='train' or row['source_id'] in heldout_sources or row['wav_sha256'] in heldout_audio
                or (row['dataset']=='google/fleurs' and row.get('sentence_id') is not None and str(row['sentence_id']) in heldout_sentences)):
            raise ValueError('TRAIN/heldout identity isolation failed')
    sampler=make_sampler(pool,config,state)
    plan=[]
    for index in range(max(args.control_updates,args.intervention_updates)):
        groups=sampler.next_batches()
        plan.append({'update':index+1,'source_ids':[[r['source_id'] for r in group] for group in groups],
            'max_frames':[max(r['encoder_frames'] for r in group) for group in groups],
            'max_labels':[max(r['target_tokens'] for r in group) for group in groups]})
    shapes={tuple(zip(row['max_frames'],row['max_labels'])) for row in plan}
    if len(shapes)<min(8,len(plan)):raise ValueError('Diverse variable shapes required')
    selected={source for step in plan for group in step['source_ids'] for source in group}
    by_id={row['source_id']:row for row in (*pool['turkish'],*pool['replay'])}
    selected_data=[]
    for source in sorted(selected):
        row=by_id[source];path=Path(row['cache_path'])
        if path.name!=hashlib.sha256(source.encode()).hexdigest()+'.pt':raise ValueError('Cache source key differs')
        info=sf.info(row['audio_filepath'])
        if info.samplerate!=16000 or info.channels!=1 or abs(info.frames/16000-row['duration'])>1e-5:
            raise ValueError('Actual TRAIN PCM duration/format differs')
        if sha(row['audio_filepath'])!=row['wav_sha256']:raise ValueError('Actual TRAIN WAV bytes differ')
        cached=torch.load(path,map_location='cpu',weights_only=True)
        if (cached['source_id']!=source or tuple(cached['features'].shape)!=(row['encoder_frames'],1024)
                or cached['labels'].numel()!=row['target_tokens'] or not torch.isfinite(cached['features']).all()):
            raise ValueError('Actual TRAIN cache metadata/finiteness differs')
        selected_data.append({'source_id':source,'wav_sha256':row['wav_sha256'],'cache_sha256':sha(path)})
    evidence={'checkpoint_path':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint),
              'source_run_id':state['run_id'],'source_step':state['step'],'optimizer_states':207,
              'run_config_sha256':sha(args.run_config),'pool_sha256':sha(args.pool),'source_sha256':config['code_sha256'],
              'probe_sha256':sha(__file__),'resource_helper_sha256':sha(OUTPUT/'resource_watchdog_v1.py'),
              'memory_launcher_v3_sha256':sha(OUTPUT/'launch_with_memory_guard_v3.py'),
              'helper_sha256':sha(OUTPUT/'cuda_cache_reclaim_v1.py'),
              'launcher_sha256':sha(OUTPUT/'launch_with_cache_reclaim_v1.py'),
              'plans':plan,'diverse_shape_count':len(shapes),'all_rows_train_and_heldout_identity_isolated':True,
              'selected_unique_clips':len(selected_data),'selected_actual_PCM_cache_hashes':selected_data,
              'CUDA_initialized_during_CPU_preparation':torch.cuda.is_initialized()}
    return config,pool,state,evidence


def make_sampler(pool,config,state):
    sampler=BalancedBucketSampler(pool['turkish'],pool['replay'],seed=config['seed'],batch_size=8,accumulation=2,
                                   replay_fraction=config['replay_fraction'],bucket_size=config['bucket_size'])
    sampler.load_state_dict(state['sampler']);return sampler


def compare_capture(reference,candidate):
    results={};all_bitwise=True
    for category in ('gradients','weights'):
        max_abs=0.;numerator=denominator=dot=other=0.;equal=True
        if set(reference[category])!=set(candidate[category]):raise ValueError('Captured trainable key mismatch')
        for name,left in reference[category].items():
            right=candidate[category][name];equal&=torch.equal(left,right)
            a,b=left.double().reshape(-1),right.double().reshape(-1);difference=a-b
            max_abs=max(max_abs,float(difference.abs().max()));numerator+=float(difference.square().sum())
            denominator+=float(a.square().sum());other+=float(b.square().sum());dot+=float((a*b).sum())
        results[category]={'bitwise_equal':equal,'max_absolute_delta':max_abs,
             'relative_l2':math.sqrt(numerator/max(denominator,1e-30)),
             'cosine':dot/max(math.sqrt(denominator*other),1e-30)};all_bitwise&=equal
    results['sampler_rng_equal']=reference['sampler_sha256']==candidate['sampler_sha256']
    results['python_cpu_cuda_rng_equal']=reference['rng_sha256']==candidate['rng_sha256']
    results['optimizer_exact_digest_equal']=reference['optimizer_sha256']==candidate['optimizer_sha256']
    results['all_bitwise_equal']=all_bitwise and all(results[key] for key in
        ('sampler_rng_equal','python_cpu_cuda_rng_equal','optimizer_exact_digest_equal'))
    results['quality_or_numerical_equivalence_waiver']=False
    return results


def run(args):
    if not args.output.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('New output must belong to the artifact workspace')
    if args.output.exists():raise FileExistsError('Use a new output; original/measured evidence cannot be overwritten')
    if not 2<=args.control_updates<=128 or not args.control_updates<=args.intervention_updates<=256:
        raise ValueError('Bounded control2..128 and intervention>=control<=256 required')
    if not 30<=args.max_wall_seconds<=720:raise ValueError('Bounded wall budget30..720sec required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    config,pool,saved,identity=read_inputs(args)
    report={'schema':1,'kind':'cuda_cache_reclaim_gpu_validation_v1','probe_version':4,'cleanup_threshold_gb':4.,'status':'prepared_not_run',
            'passed':False,'real_hook_target_passed':False,'identity':identity,'solo_gpu':args.solo_gpu,
            'scope':'TRAIN-only new harness A/B; no DEV/TEST inference, selection or original run mutation',
            'maximum_reserved_gb_guard':12.,'allocator_hard_cap_gb':9.,'maximum_commit_fraction':.95,
            'resource_policy':dict(RESOURCE_POLICY),
            'branch_policies':{'control':{'allocator_hard_cap_gb':13.,'cleanup_threshold_gb':6.},
                               'reclaim':{'allocator_hard_cap_gb':9.,'cleanup_threshold_gb':4.}},
            'actual_worker_pid':os.getpid(),'actual_worker_ppid':os.getppid(),
            'actual_worker_creation_identity':creation_identity(os.getpid()),
            'safety_cap_limit':'Control13GB and intervention9GB per-process allocator caps can trigger native autoreclaim/OOM; control is the preceding6GB-cleanup policy, not the untouched31GB fallback run. Allocation retries/OOMs and critical/sustained commit guards are recorded.',
            'harness_cleanup_policy':'After full Adam/newrow update+synchronize, control clears unused cache above6GB; intervention above4GB; timing includes cleanup.',
            'production_hook_scope':'Frozen run:539 hook has CPU real-frame proof; this harness cannot claim production-worker execution.',
            'strict_BF16_original_vs_v2_gate':'Previously failed; no threshold change or quality waiver.'}
    if args.prepare:
        write_json(args.output,report);print(json.dumps({'prepared':True,'CUDA_initialized':torch.cuda.is_initialized(),
            'diverse_shapes':identity['diverse_shape_count'],'path':str(args.output)}));return
    if not args.solo_gpu:raise ValueError('Parent-coordinated solo GPU declaration required')
    initial_sample=memory_sample('before_GPU_initialization')
    if safety_reason(initial_sample['system_memory'],0,observed_monotonic=initial_sample['observed_monotonic']):
        raise RuntimeError('System already above the new critical/telemetry guard')
    report['owned_memory_before_GPU_initialization']=initial_sample
    deadline=time.monotonic()+args.max_wall_seconds
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    capacity=torch.cuda.get_device_properties(0).total_memory
    validate_capacity(capacity,9.)
    torch.cuda.set_per_process_memory_fraction(13e9/capacity)
    model,processor,teacher,old_size,_=initialize(SimpleNamespace(weights=args.checkpoint,no_encoder_checkpoint=False))
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    adapters=[p for name,p in named.items() if name.endswith(('lora_A','lora_B'))]
    heads=[p for name,p in named.items() if not name.endswith(('lora_A','lora_B'))]
    trainable=adapters+heads
    optimizer=torch.optim.AdamW([{'params':heads,'lr':config['head_lr'],'weight_decay':.01},
                                {'params':adapters,'lr':config['lora_lr'],'weight_decay':0.}])
    new_parameters=[named[name] for name in ('decoder.embedding.weight','joint.head.weight','joint.head.bias')]
    work=args.output.parent/(args.output.stem+'_evidence');work.mkdir(exist_ok=False)
    initial_optimizer_sha=tree_sha(saved['optimizer']);captures={};matched=None
    report.update(status='running',started_at_utc=datetime.now(timezone.utc).isoformat(),
            backend_flags={'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                           'cudnn_benchmark':torch.backends.cudnn.benchmark},branches={})
    write_json(args.output,report)
    try:
        for policy,requested in [('control',args.control_updates),('reclaim',args.intervention_updates)]:
            branch_cap_gb=13. if policy=='control' else 9.
            cleanup_threshold_bytes=6_000_000_000 if policy=='control' else 4_000_000_000
            resource_guard=ResourceGuard()
            optimizer.zero_grad(set_to_none=True);clear_dp_plan_cache();torch.cuda.empty_cache()
            model.load_state_dict(saved['weights'],strict=False);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
            sampler=make_sampler(pool,config,saved)
            random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            if tree_sha(optimizer.state_dict())!=initial_optimizer_sha:raise ValueError('Exact Adam reset differs')
            if tree_sha({name:p for name,p in named.items()})!=tree_sha(saved['weights']):raise ValueError('Exact weights reset differs')
            torch.cuda.synchronize();torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(branch_cap_gb*1e9/capacity)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.memory._record_memory_history(enabled='all',context='all',stacks='python',max_entries=5000)
            initial_owned_sample=memory_sample('branch_'+policy+'_initial')
            branch={'requested_updates':requested,'updates':[],'stop_reason':None,'initial_allocator':allocator_stats(),
                    'initial_system':initial_owned_sample['system_memory'],
                    'initial_owned_memory_telemetry':initial_owned_sample,
                    'allocator_hard_cap_gb':branch_cap_gb,'cleanup_threshold_gb':cleanup_threshold_bytes/1e9,
                    'initial_driver':driver_memory(),'snapshots':[],'capture_memory_events':[]}
            report['branches'][policy]=branch
            def snapshot(label):
                raw=torch.cuda.memory._snapshot();summary=summarize_snapshot(raw)
                path=work/(policy+'_'+label+'_snapshot.json.gz')
                with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(raw,stream)
                branch['snapshots'].append({'label':label,'summary':summary,'path':str(path),'sha256':sha(path)})
            def capture():
                captured={'gradients':{name:p.grad.detach().cpu().clone() for name,p in named.items()},
                        'weights':{name:p.detach().cpu().clone() for name,p in named.items()},
                        'sampler_sha256':tree_sha(sampler.state_dict()),
                        'rng_sha256':tree_sha((random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all())),
                        'optimizer_sha256':tree_sha(optimizer.state_dict())}
                branch['capture_memory_events'].append(memory_sample('branch_'+policy+'_after_grad_weight_capture'))
                return captured
            snapshot('initial')
            for index in range(requested):
                before_sample=memory_sample('branch_'+policy+'_before_update');before_memory=before_sample['system_memory'];before_stats=allocator_stats()
                reason=safety_reason(before_memory,before_stats['reserved_bytes.all.current'],resource_guard=resource_guard,
                    observed_monotonic=before_sample['observed_monotonic'],worker_private_memory=before_sample['worker_private_memory'])
                if reason or time.monotonic()>=deadline:
                    branch['stop_reason']=reason or 'bounded_wall_deadline';break
                began=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                groups=sampler.next_batches();components=Counter();audio=0.
                for group in groups:
                    batch=collate_training_batch(processor,group,blank_id=model.config.blank_token_id,old_size=old_size,device='cuda')
                    result=batch_objective(model,teacher,batch,old_size=old_size,replay_kl=config['replay_kl'],
                        duration_kl=config['duration_kl'],feature_distillation=config['feature_distillation'],chunk_size=16,
                        tdt_loss_fn=chunked_tdt_loss,tdt_host_lengths=True)
                    scaled=result.loss/2
                    if not torch.isfinite(scaled):raise RuntimeError('Nonfinite objective')
                    scaled.backward()
                    for key,value in result.detached_metrics().items():components[key]+=value/2
                    audio+=batch.audio_seen_seconds;del result,scaled,batch
                if any(p.grad is None for p in trainable):raise RuntimeError('Missing trainable gradient')
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                previous=[p[old_size:model.config.vocab_size].detach().clone() for p in new_parameters]
                optimizer.step()
                with torch.no_grad():
                    for parameter,old in zip(new_parameters,previous):
                        changed=parameter[old_size:model.config.vocab_size]
                        changed.copy_(old+config['new_token_lr_multiplier']*(changed-old))
                torch.cuda.synchronize();pre_cleanup=allocator_stats();cleanup_seconds=0.
                if pre_cleanup['reserved_bytes.all.current']>cleanup_threshold_bytes:
                    start=time.perf_counter();torch.cuda.empty_cache();cleanup_seconds=time.perf_counter()-start
                stats=allocator_stats();seconds=time.perf_counter()-began
                post_sample=memory_sample('branch_'+policy+'_after_update_and_cache_cleanup')
                row={'update':index+1,'source_ids':[[r['source_id'] for r in group] for group in groups],
                    'max_frames':[max(r['encoder_frames'] for r in group) for group in groups],
                    'max_labels':[max(r['target_tokens'] for r in group) for group in groups],
                    'component_losses':dict(components),'gradient_norm_before_clip':float(norm),
                    'training_seconds_including_stats_cleanup':seconds,'audio_seconds':audio,
                    'pre_cleanup_allocator':pre_cleanup,'allocator':stats,'cleanup_seconds':cleanup_seconds,
                    'system':post_sample['system_memory'],'owned_memory_telemetry':post_sample,
                    'before_owned_memory_telemetry':before_sample}
                branch['updates'].append(row)
                print(json.dumps({'policy':policy,'update':index+1,'seconds':seconds,'reserved_gb':stats['reserved_bytes.all.current']/1e9,
                    'inactive_split_gb':stats['inactive_split_bytes.all.current']/1e9,'commit_fraction':row['system'].get('commit_fraction')}),flush=True)
                if index%8==0:row['driver']=driver_memory()
                if index==0:captures[(policy,1)]=capture()
                if policy=='reclaim' and index+1==matched:captures[(policy,matched)]=capture()
                reason=safety_reason(row['system'],stats['reserved_bytes.all.current'],resource_guard=resource_guard,
                    observed_monotonic=post_sample['observed_monotonic'],worker_private_memory=post_sample['worker_private_memory'])
                row['moderate_commit_streak_after_observation']=resource_guard.moderate_streak
                if reason:branch['stop_reason']=reason;break
            if not branch['updates']:raise RuntimeError('Safety guard allowed no trained updates')
            completed=len(branch['updates'])
            if policy=='control':matched=completed;captures[(policy,matched)]=capture()
            if policy=='reclaim' and completed<matched:raise RuntimeError('Intervention did not reach shared prefix')
            snapshot('final');torch.cuda.memory._record_memory_history(enabled=None)
            final_owned_sample=memory_sample('branch_'+policy+'_final')
            branch.update(completed_updates=completed,final_driver=driver_memory(),final_system=final_owned_sample['system_memory'],
                    final_owned_memory_telemetry=final_owned_sample,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    actual_audio_min_per_wall_min=sum(row['audio_seconds'] for row in branch['updates'])/
                        sum(row['training_seconds_including_stats_cleanup'] for row in branch['updates']))
            if tree_sha(saved['optimizer'])!=initial_optimizer_sha:raise RuntimeError('Immutable baseline optimizer was mutated')
            write_json(args.output,report)
        comparisons={str(step):compare_capture(captures[('control',step)],captures[('reclaim',step)]) for step in sorted({1,matched})}
        equal_sources=all(a['source_ids']==b['source_ids'] for a,b in zip(report['branches']['control']['updates'],report['branches']['reclaim']['updates']))
        max_loss_delta=max(abs(a['component_losses'][key]-b['component_losses'][key]) for a,b in
            zip(report['branches']['control']['updates'],report['branches']['reclaim']['updates']) for key in a['component_losses'])
        artifact=work/'matched_prefix_gradients_weights.pt';torch.save(captures,artifact)
        report.update(status='complete',passed=equal_sources and all(c['sampler_rng_equal'] and c['python_cpu_cuda_rng_equal'] for c in comparisons.values()),
            actual_trained_updates=sum(b['completed_updates'] for b in report['branches'].values()),
            shared_prefix_updates=matched,matched_sources_equal=equal_sources,max_component_absolute_delta=max_loss_delta,
            comparisons=comparisons,numerical_differences_explicit=True,full_capture_path=str(artifact),full_capture_sha256=sha(artifact),
            production_hook_validation_pending=True,
            actual_self_PID_private_telemetry_recorded=True,
            real_hook_target_passed=False,
            interpretation='Inactive allocator bytes and cleanup response support/cache hypothesis only if observed; GPU paging traffic is not measured.')
        report['intervention_cleanup_calls']=sum(row['cleanup_seconds']>0 for row in report['branches']['reclaim']['updates'])
        report['intervention_requested_updates_completed']=report['branches']['reclaim']['completed_updates']==args.intervention_updates
        report['matched_prefix_timing_caveat']='Control runs first; process kernel caches may benefit the later intervention. Memory-state/cleanup evidence is primary; this single ordering is not an unbiased throughput benchmark.'
        if (sha(args.run_config)!=identity['run_config_sha256'] or sha(args.pool)!=identity['pool_sha256']
                or sha(args.checkpoint)!=identity['checkpoint_sha256']
                or any(sha(OUTPUT/name)!=digest for name,digest in config['code_sha256'].items())):
            raise RuntimeError('Original input/source/weights changed during the probe')
    except BaseException as error:
        report.update(status='failed',passed=False,failure_type=type(error).__name__,failure_message=str(error));raise
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),probe_sha256_after=sha(__file__),
            checkpoint_sha256_after=sha(args.checkpoint),original_run_mutated=False)
        write_json(args.output,report)
        optimizer.zero_grad(set_to_none=True);clear_dp_plan_cache();torch.cuda.empty_cache()


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    run_id='adaptive_20261010_continuation_02_resume_B8'
    value.add_argument('--checkpoint',type=Path,default=ROOT/'work'/f'{run_id}_latest.pt')
    value.add_argument('--run-config',type=Path,default=OUTPUT/'runs'/run_id/'run_config.json')
    value.add_argument('--pool',type=Path,default=OUTPUT/'runs'/run_id/'training_pool.json')
    value.add_argument('--output',type=Path,default=OUTPUT/'cuda_cache_reclaim_gpu_probe_v4.json')
    value.add_argument('--control-updates',type=int,default=10)
    value.add_argument('--intervention-updates',type=int,default=100)
    value.add_argument('--max-wall-seconds',type=float,default=600.)
    value.add_argument('--prepare',action='store_true')
    value.add_argument('--solo-gpu',action='store_true')
    return value


if __name__=='__main__':run(parser().parse_args())
