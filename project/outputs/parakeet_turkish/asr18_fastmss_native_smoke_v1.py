"""Bounded native TRAIN audio-reader proof; no production/quality trial."""
import gc
import ctypes
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from atomic_io_retry_v1 import write_json
from asr18_diagnostic_bootstrap_v1 import AREA,STORAGE,record,require
from resource_watchdog_v2 import sample_memory


def digest(tensor):
    return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def rng_identity():
    numpy=np.random.get_state()
    return {'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            'torch':digest(torch.get_rng_state()),'cuda':[digest(state) for state in torch.cuda.get_rng_state_all()],
            'numpy':hashlib.sha256(numpy[1].tobytes()+repr((numpy[0],*numpy[2:])).encode()).hexdigest()}


def logical_IO():
    class Counters(ctypes.Structure):
        _fields_=[(name,ctypes.c_ulonglong) for name in ('read_operations','write_operations','other_operations','read_bytes','write_bytes','other_bytes')]
    value=Counters();api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetProcessIoCounters.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters)]
    require(api.GetProcessIoCounters(ctypes.c_void_p(-1),ctypes.byref(value)),'Owned process I/O counters unavailable')
    return {name:int(getattr(value,name)) for name,_ in value._fields_}


def validate(model,processor,teacher,optimizer,provenance,folder,*,stage,admit):
    from fastmss_train_augmentation_v1 import AugmentationPolicy,TrainWaveformAugmenter
    from asr18_diagnostic_bootstrap_v1 import PROJECT
    from tsc_bpe_full_decoder_transition_v1 import preflight
    from train_lora_batched_tsc_bpe_full_decoder_v1 import load_catalogs
    from bpe_training_label_catalog_v1 import BPETrainingLabels
    from batched_lora_tr_metadata_v1 import collate_training_batch
    from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
    from TR_tdt_population_weighting_v1 import apply_to_objective
    from native_tdt_checkpointed_v4 import chunked_tdt_loss,reclaim_inactive_cuda_cache
    from runtime import read_audio
    from functools import partial
    evidence=STORAGE/'analysis/FastMSS_integration_v1/FastMSS_integration_readiness_v1.json'
    readiness=json.loads(evidence.read_text())
    adapter=PROJECT/'outputs/parakeet_turkish/fastmss_train_augmentation_v1.py'
    require(record(adapter) in readiness['touched_files'],'Pinned CPU-proved adapter required')
    pilot=adapter.with_name('fastmss_train_augmentation_v1.pilot.disabled.json')
    policy_json=json.loads(pilot.read_text())
    policy_json.update(enabled=True,apply_probability=1.,noise_probability=1.,rir_probability=1.)
    policy_path=folder/'enabled_diagnostic_policy.json';write_json(policy_path,policy_json)
    policy=AugmentationPolicy.from_json(policy_path)
    require(policy.noise_mode=='synthetic_gaussian','Explicit prepared synthetic-noise pilot required')
    maximum={'private_bytes':0}
    def check():
        require(not (folder/'stop.json').exists(),'Parent resource stop during bounded augmentation proof')
        sample=sample_memory(os.getpid(),'native_augmentation_callback')
        maximum['private_bytes']=max(maximum['private_bytes'],sample['worker_private_memory']['private_bytes'])
    augmenter=TrainWaveformAugmenter(policy,guard=check)
    journal=folder/'augmentation_draws.jsonl'
    def append(receipt):
        with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps(receipt)+'\n')
    config=json.loads((STORAGE/'runs/adaptive_20261015_TR_BPE512_full_decoder_02/run_config.json').read_text())
    context=preflight(SimpleNamespace(**config['arguments']));original,tr,replay=load_catalogs(context)
    labels=BPETrainingLabels(context['BPE_label_directory'],tr)
    manifest=json.loads((AREA/'control_manifest_v3/manifest.json').read_text())
    controls=[manifest['rows'][i] for i in (0,1,14,15)]
    require([row['language'] for row in controls]==['tr','tr','pt','en'],'Declared mixed TRAIN smoke controls required')
    native=partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1_700_000_000)
    position={'draw_index':0}
    def context_for(row):
        return {'split':'train','phase':'weak_label_continued_training','role':'student_tr' if row['language']=='tr' else 'replay',
                'epoch':0,'draw_index':position['draw_index']}
    read=augmenter.audio_reader(read_audio,context_for=context_for,record=append)
    waveform_checks=[]
    def observed_reader(row):
        before_rng=rng_identity();before=sample_memory(os.getpid(),'before_native_augmentation_reader')
        io0=logical_IO();clean_began=time.perf_counter();clean=read_audio(row);clean_read_seconds=time.perf_counter()-clean_began;io1=logical_IO()
        began=time.perf_counter();cpu_began=time.process_time();augmented=read(row)
        reader_seconds=time.perf_counter()-began;reader_CPU_seconds=time.process_time()-cpu_began;io2=logical_IO()
        require(before_rng==rng_identity(),'Audio adapter changed global Python/NumPy/Torch/CUDA RNG')
        require(clean.shape==augmented.shape and clean.dtype==augmented.dtype==np.float32,'Exact waveform geometry required')
        same=np.array_equal(clean,augmented)
        require(same if row['language']!='tr' else not same,'TR must augment and replay must remain byte exact')
        fresh=TrainWaveformAugmenter(policy,guard=check)
        reproduced,_=fresh.apply(clean,row,**context_for(row))
        require(np.array_equal(augmented,reproduced),'Fresh-instance durable draw replay differs in native process')
        after=sample_memory(os.getpid(),'after_native_augmentation_reader')
        waveform_checks.append({'source_id':row['source_id'],'language':row['language'],'samples':clean.size,
            'clean_equal':same,'fresh_instance_byte_exact':True,'global_RNG_unchanged':True,
            'clean_read_seconds':clean_read_seconds,'augmented_reader_seconds':reader_seconds,
            'reader_CPU_seconds':reader_CPU_seconds,'resume_proof_seconds':time.perf_counter()-began-reader_seconds,
            'clean_reader_logical_IO_delta':{key:io1[key]-io0[key] for key in io0},
            'augmented_reader_logical_IO_delta':{key:io2[key]-io1[key] for key in io0},
            'clean_waveform_sha256':hashlib.sha256(clean.tobytes()).hexdigest(),'augmented_waveform_sha256':hashlib.sha256(augmented.tobytes()).hexdigest(),
            'private_before':before['worker_private_memory']['private_bytes'],
            'private_after':after['worker_private_memory']['private_bytes']})
        return augmented
    def geometry(batch):
        return {'labels':digest(batch.labels),'decoder_ids':digest(batch.decoder_input_ids),
                'target_lengths':batch.target_lengths.tolist(),'frame_lengths':batch.frame_lengths.tolist(),
                'attention_mask':digest(batch.inputs['attention_mask']),'input_features':digest(batch.inputs['input_features']),
                'replay_features':digest(batch.replay_base_features),'replay_indices':list(batch.replay_indices)}
    def fingerprints():return {name:digest(parameter) for name,parameter in optimizer.parameters.items()}
    random_state=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    clean_geometry=[];clean_losses=[];augmented_losses=[];comparisons=[]
    model.train();model.encoder.eval()
    try:
        admit('native_update_augmentation_reference',provenance['minimal_native_import']['cold_native_budget_bytes'])
        for augmented_phase in (False,True):
            optimizer.zero_grad()
            random.setstate(random_state[0]);torch.set_rng_state(random_state[1]);torch.cuda.set_rng_state_all(random_state[2])
            started=time.monotonic()
            for index,row in enumerate(controls):
                check();position['draw_index']=index;rewritten=labels.rewrite_batch([row])
                batch=collate_training_batch(processor,rewritten,blank_id=8192,old_size=8193,device='cuda',
                    audio_reader=observed_reader if augmented_phase else read_audio)
                actual=geometry(batch)
                if augmented_phase:
                    expected=clean_geometry[index]
                    require(all(actual[key]==expected[key] for key in actual if key!='input_features'),'Targets/masks/frame/replay cache changed')
                    require(actual['input_features']==expected['input_features'] if row['language']!='tr' else
                            actual['input_features']!=expected['input_features'],'Native mel feature protection/augmentation failed')
                    comparisons.append({'source_id':row['source_id'],'labels_lengths_masks_frames_replay_cache_exact':True,
                        'replay_input_features_exact':row['language']!='tr','TR_input_features_changed':row['language']=='tr'})
                else:clean_geometry.append(actual)
                result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40.,duration_kl=2.,
                    feature_distillation=20.,portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
                if row['language']=='tr':result,_=apply_to_objective(result,rewritten,mode='token_population',population_mean_U=26.642061797177284,torch_module=torch)
                require(bool(torch.isfinite(result.total)),'Finite augmented/clean native objective required')
                (result.total/4).backward()
                (augmented_losses if augmented_phase else clean_losses).append(float(result.total.detach()))
                del batch,result,rewritten;reclaim_inactive_cuda_cache(torch.device('cuda'),'augmentation_smoke_after_clip')
                stage('native_augmentation_microbatch_complete',augmented=augmented_phase,microbatch=index+1)
            stage('native_augmentation_phase_complete',augmented=augmented_phase,seconds=time.monotonic()-started)
            if not augmented_phase:optimizer.zero_grad();torch.cuda.empty_cache()
        before=fingerprints();norm=optimizer.step();optimizer.zero_grad();torch.cuda.synchronize();torch.cuda.empty_cache()
        after=fingerprints();changed=[name for name in before if before[name]!=after[name]]
        require(any(name in optimizer.head_names for name in changed) and any(name in optimizer.lora_names for name in changed),'Positive real head+LoRA update required')
        stage('native_augmentation_positive_Adam_update',clocks=[64251,2001,62251],global_grad_norm=float(norm))
        admit('native_augmentation_durable_checkpoint',300_000_000)
        next_context={**context_for(controls[0]),'draw_index':4}
        next_waveform,_=augmenter.apply(read_audio(controls[0]),controls[0],**next_context)
        next_draw={'source_id':controls[0]['source_id'],'context':next_context,
                   'waveform_sha256':hashlib.sha256(next_waveform.tobytes()).hexdigest()}
        del next_waveform
        checkpoint=folder/'augmented_native_update1.pt';temporary=folder/'augmented_native_update1.writing.pt'
        payload={'kind':'ASR18_FastMSS_native_smoke_v1','diagnostic_only':True,'weights':{name:p.detach().cpu() for name,p in optimizer.parameters.items()},
            'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'augmentation':{'policy':record(policy_path),'policy_sha256':augmenter.policy_sha256,'registry_sha256':augmenter.registry_sha256,
                            'draw_receipts':record(journal),'completed_smoke_updates':1,'next_diagnostic_draw_index':4,
                            'next_draw_identity':next_draw},'source_provenance':provenance}
        torch.save(payload,temporary)
        with temporary.open('r+b') as stream:os.fsync(stream.fileno())
        temporary.replace(checkpoint);del payload;gc.collect()
        admit('native_augmentation_checkpoint_and_draw_restore',350_000_000)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);before_restore=fingerprints()
        model.load_state_dict(saved['weights'],strict=False);optimizer.load_state_dict(saved['optimizer'])
        from bpe_broad_full207_optimizer_v1 import bits_equal
        for actual_opt,old_opt in zip(optimizer.optimizers,saved['optimizer']['optimizers']):
            for index,parameter in enumerate(actual_opt.param_groups[0]['params']):
                require(all(bits_equal(torch,actual_opt.state[parameter][key],old_opt['state'][index][key])
                    for key in ('step','exp_avg','exp_avg_sq')),'Augmented native moment/clock restore differs')
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        require(fingerprints()==before_restore and optimizer.updates==1,'Augmented native weight/clock restore differs')
        restored_policy=AugmentationPolicy.from_json(saved['augmentation']['policy']['path'])
        resumed=TrainWaveformAugmenter(restored_policy,guard=check)
        require(resumed.policy_sha256==saved['augmentation']['policy_sha256'] and resumed.registry_sha256==saved['augmentation']['registry_sha256'],
                'Resumed augmentation policy/registry differs')
        resumed_waveform,_=resumed.apply(read_audio(controls[0]),controls[0],**saved['augmentation']['next_draw_identity']['context'])
        require(hashlib.sha256(resumed_waveform.tobytes()).hexdigest()==saved['augmentation']['next_draw_identity']['waveform_sha256'],
                'Actual native checkpoint next augmentation draw differs')
        del saved,resumed_waveform;gc.collect()
        stage('native_augmentation_checkpoint_and_next_draw_restore_byte_exact')
        report={'passed':True,'actual_positive_Adam_updates':1,'all207_gradients_required_by_optimizer':True,
            'head_tensors_changed':sum(name in optimizer.head_names for name in changed),'LoRA_tensors_changed':sum(name in optimizer.lora_names for name in changed),
            'clocks':[64251,2001,62251],'physical_batch':1,'accumulation':4,'controls':[row['source_id'] for row in controls],
            'policy':record(policy_path),'adapter':record(adapter),'CPU_readiness':record(evidence),'registry_sha256':augmenter.registry_sha256,
            'draw_receipts':record(journal),'waveform_checks':waveform_checks,'native_geometry_checks':comparisons,
            'clean_losses':clean_losses,'augmented_losses':augmented_losses,'global_grad_norm':float(norm),'callback_private_peak_bytes':maximum['private_bytes'],
            'checkpoint':record(checkpoint),'clean_reference_has_no_Adam_update':True,'warmup_order':'clean first; augmented second; not a cold startup overhead estimate',
            'actual_checkpoint_weights_Adam_steps_and_next_draw_restore_byte_exact':True,
            'I_O_counter_scope':'Windows owned-process logical I/O; not physical storage bytes; journal writes included',
            'DEV_TEST_clean_unchanged':True,'quality_or_production_augmentation_enabled_claimed':False,
            'noise_is_synthetic_not_environmental_recording':True,'whole_FastMSS_meeting_recipe_used':False}
        write_json(folder/'FastMSS_native_validation.json',report)
        return {'passed':True,'positive_native_update':True,'real_native_updates':1,'FastMSS_native_smoke_passed':True,'quality_gain_claimed':False}
    finally:
        labels.close()
        if tr is not original:tr.close()
        original.close()
