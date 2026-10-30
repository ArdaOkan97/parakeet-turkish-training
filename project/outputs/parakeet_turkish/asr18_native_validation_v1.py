"""Bounded real-update/DEV/checkpoint memory validation; no scientific winner."""
import gc
import hashlib
import json
import os
import random
import time
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import torch
from atomic_io_retry_v1 import write_json
from asr18_diagnostic_bootstrap_v1 import AREA,STORAGE,record,require
from resource_watchdog_v2 import sample_memory

class ControlledStop(Exception): pass

def validate(model,processor,teacher,optimizer,provenance,folder,*,stage,admit,cached_teacher=None):
    from tsc_bpe_full_decoder_transition_v1 import preflight,validate_scores
    from train_lora_batched_tsc_bpe_full_decoder_v1 import load_catalogs,evaluate,gate
    from bpe_training_label_catalog_v1 import BPETrainingLabels
    from batched_lora_tr_metadata_v1 import collate_training_batch
    from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
    from TR_tdt_population_weighting_v1 import apply_to_objective
    from native_tdt_checkpointed_v4 import chunked_tdt_loss,reclaim_inactive_cuda_cache,cache_cleanup_evidence
    import benchmark
    config=json.loads((STORAGE/'runs/adaptive_20261015_TR_BPE512_full_decoder_02/run_config.json').read_text())
    context=preflight(SimpleNamespace(**config['arguments']))
    original,tr,replay=load_catalogs(context)
    labels=BPETrainingLabels(context['BPE_label_directory'],tr)
    updates=20;rows=[]
    for update in range(updates):
        part=[tr[((14*update+j)*25198)%len(tr)] for j in range(14)]
        for language in ('pt',('en','es','it','fr','de')[update%5]):
            language_rows=[r for r in replay if r['language']==language]
            part.append(language_rows[update%len(language_rows)])
        rows.append(part)
    # Repeat a fixed, declared heterogeneous workload after its first measured
    # warmup: new shape families cannot silently invalidate a learned peak.
    control_report=json.loads((AREA/'control_manifest_v3/report.json').read_text())
    require(control_report['passed'] and not control_report['GPU_used'] and not control_report['ML_runtime_imported'],
            'Actual CPU-only control preflight required')
    require(record(control_report['manifest']['path'])==control_report['manifest'],'Pinned control metadata changed')
    control_manifest=json.loads(Path(control_report['manifest']['path']).read_text())
    require(control_manifest['source_training_pool']==context['source_training_pool'],'Control TRAIN source differs')
    control=control_manifest['rows']
    require(all(tr[ordinal]==row for ordinal,row in zip(control_manifest['TR_ordinals'],control[:14]))
            and all(any(row==saved for saved in replay) for row in control[14:]),'Control metadata not exact source rows')
    control_shapes=[]
    for row in control:
        rewritten=labels.rewrite_batch([row])[0]
        joint=rewritten['encoder_frames']*(rewritten['target_tokens']+1)*8716*4
        control_shapes.append({'source_id':row['source_id'],'frames':rewritten['encoder_frames'],
                               'tokens':rewritten['target_tokens'],'full_joint_bytes':joint})
    require(max(item['full_joint_bytes'] for item in control_shapes)<=40_000_000,
            'Declared CPU-prevalidated40MB control exceeded; no clip/crop/filter fallback')
    rows=[control for _ in range(updates)]
    declaration={'kind':'ASR18_bounded_memory_diagnostic_fork_v1','scientific_training_started':False,
        'updates_bound':updates,'effective_clips_per_update':16,'TR_PT_oldfive':[14,1,1],
        'physical_batch':1,'accumulation':16,'one_global_clip_and_Adam_step_per16':True,
        'physical_batch8_accumulation2_bitwise_update_equivalence_claimed':False,
        'diagnostic_pool':'CPU-prevalidated14TR 8-16MB joints plus shortest-geometry PT/EN TRAIN controls; repeated20times; not broad-stage epoch',
        'control_manifest':control_report['manifest'],'control_CPU_preflight':record(AREA/'control_manifest_v3/report.json'),
        'control_geometry':control_shapes,'training_pool_cutoff_or_quality_filter_introduced':False,
        'full_joint_ceiling_unchanged':1_700_000_000,'full_large_geometry_training_fit_claimed':False,
        'replay_old_language':'en','inactive_CUDA_cache_reclaimed_after_every_full_update':True,
        'TR_population_denominator':26.642061797177284,'source_histories_immutable':True,
        'qualified_winner_replacement_authorized_or_performed':False,
        'rows':[[{'source_id':r['source_id'],'language':r['language'],'duration':r['duration'],
                   'sampling_target_tokens':r['target_tokens'],'encoder_frames':r['encoder_frames']} for r in part] for part in rows]}
    write_json(folder/'diagnostic_declaration.json',declaration)
    history=[];devs={};completed=0;stop_reason=None;restore_proved=False
    native=partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1_700_000_000)
    def check_stop():
        if (folder/'stop.json').exists(): raise ControlledStop(json.loads((folder/'stop.json').read_text())['reason'])
    def fingerprints():
        return {n:hashlib.sha256(memoryview(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()
                for n,p in optimizer.parameters.items()}
    def checkpoint(label):
        optimizer.zero_grad();torch.cuda.empty_cache();admit('checkpoint_'+label,300_000_000)
        stage('before_checkpoint_'+label)
        path=folder/(label+'.pt');temporary=folder/(label+'.writing.pt')
        payload={'kind':'ASR18_memory_diagnostic_checkpoint_v1','diagnostic_only':True,
            'weights':{n:p.detach().cpu() for n,p in optimizer.parameters.items()},'optimizer':optimizer.state_dict(),
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'diagnostic_updates':completed,'history':history,'source_provenance':provenance,'declaration':declaration}
        torch.save(payload,temporary)
        with temporary.open('r+b') as stream:os.fsync(stream.fileno())
        temporary.replace(path);del payload;gc.collect()
        stage('after_checkpoint_'+label)
        return path
    def dev(label):
        optimizer.zero_grad();torch.cuda.empty_cache();admit('full350_DEV_'+label,1_600_000_000 if label=='streamed_baseline' else 700_000_000)
        stage('before_full350_DEV_'+label)
        previous=benchmark.inputs_for
        def observed_inputs(processor,row):
            check_stop();return previous(processor,row)
        benchmark.inputs_for=observed_inputs
        try: scores=evaluate(model,processor,'ASR18_memory_'+folder.name+'_'+label)
        finally: benchmark.inputs_for=previous
        validate_scores(scores);devs[label]=gate(scores)
        stage('after_full350_DEV_'+label,DEV=devs[label]);torch.cuda.empty_cache()
        return scores
    try:
        stage('controlled_native_dataset_ready',physical_batch=1,accumulation=16)
        checkpoint('initial_full207')
        streamed=hasattr(model,'_diagnostic_frozen_backing')
        previous=AREA/'native_mixed_memory_validation_01/baseline_source_equivalence.json'
        previous_metadata=STORAGE/'results/ASR18_memory_native_mixed_memory_validation_01_baseline/metadata.json'
        previous_summary=previous_metadata.with_name('summary.json')
        proof=json.loads(previous.read_text());metadata=json.loads(previous_metadata.read_text())
        fingerprint=benchmark.model_fingerprint(model,processor)
        baseline_reusable=proof['passed'] and fingerprint==metadata['model_fingerprint']
        if streamed and not baseline_reusable:
            initial=dev('streamed_baseline');devs['baseline']=gate(initial)
            write_json(folder/'fresh_streamed_baseline_proof.json',{'passed':True,'newbaseline350decodes_performed':True,
                'fingerprint':benchmark.model_fingerprint(model,processor),'frozen_matrix_metrics':model._diagnostic_frozen_metrics})
        else:
            require(baseline_reusable,
                    'Reuse350DEV only after exact full model/tokenizer/config/code fingerprint equality')
            initial=json.loads(previous_summary.read_text());devs['baseline']=gate(initial)
            write_json(folder/'baseline_reuse_proof.json',{'passed':True,'previous_equivalence':record(previous),
                'metadata':record(previous_metadata),'summary':record(previous_summary),'full_fingerprint_equal':True,
                'newbaseline350decodes_performed':False})
            stage('exact_fingerprint_350baseline_reused',DEV=devs['baseline'])
        source_summary=STORAGE/'results/adaptive_20261015_TR_BPE512_full_decoder_02_step_64250/summary.json'
        if source_summary.exists():
            prior=json.loads(source_summary.read_text())
            require(all(initial['validation_'+l]['asr']['word_errors']==prior['validation_'+l]['asr']['word_errors']
                        for l in ('en','es','it','fr','de','tr','pt')),'Direct bootstrap baseline differs from saved BPE source DEV')
            write_json(folder/'baseline_source_equivalence.json',{'passed':True,'saved_source_DEV':record(source_summary),'all7_word_errors_equal':True})
        model.train();model.encoder.eval()
        measured_peak=None
        phase_peak={'private':sample_memory(os.getpid(),'native_phase_peak_start')['worker_private_memory']['private_bytes'],'enabled':True}
        peak_stop=threading.Event()
        def observe_native_peak():
            while not peak_stop.wait(.05):
                sample=sample_memory(os.getpid(),'native_phase_private_peak')['worker_private_memory']
                if phase_peak['enabled'] and sample['available']:phase_peak['private']=max(phase_peak['private'],sample['private_bytes'])
        peak_thread=threading.Thread(target=observe_native_peak,name='owned-native-memory-telemetry',daemon=True)
        peak_thread.start()
        for update,part in enumerate(rows,1):
            check_stop()
            current=sample_memory(os.getpid(),'native_repeated_shape_admission')['worker_private_memory']['private_bytes']
            cold_budget=provenance.get('minimal_native_import',{}).get('cold_native_budget_bytes',2_000_000_000)
            budget=cold_budget if measured_peak is None else max(100_000_000,measured_peak-current+100_000_000)
            admit('native_update_'+str(update),budget)
            optimizer.zero_grad();began=time.monotonic();losses=[]
            rng=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            before=fingerprints() if update<=2 else None
            try:
                for index,row in enumerate(part):
                    check_stop();rewritten=labels.rewrite_batch([row])
                    batch=collate_training_batch(processor,rewritten,blank_id=8192,old_size=8193,device='cuda')
                    if cached_teacher is not None and row['language']!='tr':cached_teacher.activate(row['source_id'])
                    result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40.,duration_kl=2.,
                        feature_distillation=20.,portuguese_token_kl_multiplier=1.,chunk_size=16,
                        tdt_loss_fn=native,tdt_host_lengths=True)
                    if row['language']=='tr':
                        result,_=apply_to_objective(result,rewritten,mode='token_population',
                                                    population_mean_U=26.642061797177284,torch_module=torch)
                    require(bool(torch.isfinite(result.total)),'Nonfinite real native objective')
                    (result.total/16).backward();losses.append(float(result.total.detach()))
                    del result,batch,rewritten
                    reclaim_inactive_cuda_cache(torch.device('cuda'),'diagnostic_after_single_clip_backward')
                    stage('native_microbatch_complete',diagnostic_update=update,microbatch=index+1)
            except ControlledStop:
                optimizer.zero_grad();random.setstate(rng[0]);torch.set_rng_state(rng[1]);torch.cuda.set_rng_state_all(rng[2])
                torch.cuda.empty_cache();raise
            norm=optimizer.step();optimizer.zero_grad();torch.cuda.synchronize()
            torch.cuda.empty_cache()
            completed=update;row={'diagnostic_update':update,'clocks':[c+update for c in (64250,2000,62250)],
                'mean_clip_objective':sum(losses)/16,'global_grad_norm':float(norm),'seconds':time.monotonic()-began}
            if before is not None:
                after=fingerprints();changed=[n for n in before if before[n]!=after[n]]
                row.update(changed_head_tensors=sum(n in optimizer.head_names for n in changed),
                           changed_encoder_LoRA_tensors=sum(n in optimizer.lora_names for n in changed))
                require(row['changed_head_tensors']>0 and row['changed_encoder_LoRA_tensors']>0,'Real decoder and LoRA weight changes required')
            history.append(row);write_json(folder/'native_progress.json',{'completed_updates':completed,'latest':row,'history':history})
            observation=stage('native_optimizer_update_complete',**row)
            measured_peak=max(phase_peak['private'],observation['worker_private_memory']['private_bytes'])
            row['native_phase_private_peak_50ms_bytes']=measured_peak
            row['OS_private_peak_bytes_lifetime']=observation['worker_private_peak_bytes_lifetime']
            if update==19:
                phase_peak['enabled']=False
                path=checkpoint('after_native_update19');admit('native_checkpoint_restore',350_000_000)
                saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True);before=fingerprints()
                model.load_state_dict(saved['weights'],strict=False);optimizer.load_state_dict(saved['optimizer'])
                from bpe_broad_full207_optimizer_v1 import bits_equal
                for actual_opt,old_opt in zip(optimizer.optimizers,saved['optimizer']['optimizers']):
                    for index,parameter in enumerate(actual_opt.param_groups[0]['params']):
                        require(all(bits_equal(torch,actual_opt.state[parameter][key],old_opt['state'][index][key])
                                    for key in ('step','exp_avg','exp_avg_sq')),'Real native Adam moment restore not byte exact')
                random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
                require(before==fingerprints() and optimizer.updates==19,'Real native checkpoint restore changed exact state')
                del saved;gc.collect();restore_proved=True
                stage('native_checkpoint_restore_byte_exact',clocks=[64269,2019,62269])
                phase_peak['enabled']=True
        final_path=checkpoint('after_native_update20')
        write_json(folder/'native_training_validation.json',{'passed':True,'completed_updates':completed,'history':history,'DEV':devs,
            'checkpoint':record(final_path),'native_checkpoint_restore_byte_exact':restore_proved,
            'native_inactive_cache_evidence':cache_cleanup_evidence(),'quality_gain_or_winner_claimed':False,
            'physical_batch':1,'accumulation':16,'optimizer_clocks':[64270,2020,62270],
            'final_DEV_requires_fresh_serial_optimizer_free_child':True,'large_geometry_fit_proved':False})
        return {'passed':True,'positive_native_update':True,'real_native_updates':completed,'full_DEV_results':len(devs),
                'new_full350_DEV_in_this_worker':0 if baseline_reusable else 1,'reused_full350_source_equivalent_baseline':baseline_reusable,'final_DEV_pending':True,
                'checkpoint_restore_proved':restore_proved,'quality_gain_claimed':False}
    except ControlledStop as exc:
        stop_reason=str(exc);stage('cooperative_diagnostic_stop',reason=stop_reason,completed_updates=completed)
        optimizer.zero_grad();torch.cuda.empty_cache()
        # A previously committed checkpoint remains valid even when no new save
        # can be admitted under current external system pressure.
        write_json(folder/'native_partial.json',{'passed':False,'completed_updates':completed,'history':history,
                   'DEV':devs,'stop_reason':stop_reason,'partial_update_discarded_with_RNG_restored':True})
        return {'passed':False,'positive_native_update':completed>0,'real_native_updates':completed,'stop_reason':stop_reason}
    finally:
        if 'peak_stop' in locals():
            peak_stop.set();peak_thread.join(timeout=2)
        labels.close()
        if tr is not original:tr.close()
        original.close()
