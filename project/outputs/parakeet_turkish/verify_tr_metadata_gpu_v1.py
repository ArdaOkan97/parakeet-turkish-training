"""Root-owned bounded TRAIN-only GPU mask proof, no caches/training/selection."""
import argparse,hashlib,json,time,math,os
from pathlib import Path
from datetime import datetime,timezone
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
MANIFEST_SHA='d2d0f6df539b1e5cfe9488960a5859bc6ab1870a1ffc634b5b707bc2cc2f5704'
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--root-owned-solo-gpu',action='store_true');return p
def main(args):
    if not args.root_owned_solo_gpu or args.report.exists() or not args.report.resolve().is_relative_to(ROOT):raise ValueError('Root-owned soleGPU and NEW workspace evidence required')
    manifest=ROOT/'work/data/tr_public_v3_train.jsonl';metadata_path=ROOT/'work/data/tr_public_v3_frame_metadata_v1.jsonl'
    proof_path=D/'TR_frame_metadata_cpu_verification_v1.json';proof=json.loads(proof_path.read_text())
    if not proof['passed'] or sha(manifest)!=MANIFEST_SHA or sha(metadata_path)!=proof['metadata_sha256']:raise ValueError('Exact completed CPU metadata proof required')
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line]
    ordered=sorted(rows,key=lambda r:(r['duration'],r['source_id']))
    selected=[ordered[round(i*(len(ordered)-1)/15)] for i in range(16)]
    metadata={row['source_id']:row for row in [json.loads(line) for line in metadata_path.read_text(encoding='utf-8').splitlines() if line]}
    sources={name:sha(D/name) for name in ['runtime.py','batched_lora.py','batched_lora_tr_metadata_v1.py','tr_frame_metadata_v1.py','train_lora_batched_data_v5.py','resource_watchdog_v2.py']}
    from atomic_io_retry_v1 import write_json
    started=time.perf_counter()
    report={'schema':1,'kind':'TR_metadata_actual_GPU_masks_v1','status':'running','passed':False,'TRAIN_only':True,
        'root_asserted_solo_gpu':True,'helper_sha256':sha(__file__),'source_sha256':sources,'manifest_sha256':MANIFEST_SHA,
        'CPU_frame_proof_sha256':sha(proof_path),'metadata_sha256':proof['metadata_sha256'],'physical_batch':8,
        'allocator_cap_bytes':9_000_000_000,'cleanup_threshold_bytes':4_000_000_000,'source_ids':[r['source_id'] for r in selected],
        'training_updates':0,'cache_files_written':0,'DEV_or_TEST_predictions_read':False,'numeric_equivalence_claimed':False}
    write_json(args.report,report)
    try:
        import torch,soundfile as sf
        torch.set_num_threads(2)
        from runtime import load_model
        total=torch.cuda.get_device_properties(0).total_memory
        if not 15e9<=total<=18e9:raise ValueError('Reviewed16GB-class device required')
        torch.cuda.set_per_process_memory_fraction(9e9/total,device=0)
        model,processor=load_model();model.eval().requires_grad_(False)
        from resource_watchdog_v2 import sample_memory,ResourceGuard,worker_process_commit_reason
        guard=ResourceGuard();checks=[]
        for offset in [0,8]:
            group=selected[offset:offset+8];audio=[]
            for row in group:
                if sha(row['audio_filepath'])!=row['wav_sha256']:raise ValueError('TRAIN WAV content changed')
                value,sr=sf.read(row['audio_filepath'],dtype='float32')
                if sr!=16000 or value.ndim!=1 or len(value)!=metadata[row['source_id']]['actual_PCM_frames']:raise ValueError('Actual TRAIN audio lengths changed')
                audio.append(value)
            inputs=dict(processor(audio=audio,sampling_rate=16000,return_tensors='pt',padding='longest'))
            masks=inputs['attention_mask'].bool();input_lengths=masks.sum(-1).tolist()
            if input_lengths!=[metadata[r['source_id']]['valid_mel_frames'] for r in group]:raise ValueError('Actual input masks differ from CPU derived metadata')
            inputs['input_features']=inputs['input_features'].masked_fill(~masks.unsqueeze(-1),0)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model.encoder(**{name:value.cuda() for name,value in inputs.items()})
            actual=out.attention_mask.sum(-1).cpu().tolist();expected=[metadata[r['source_id']]['encoder_frames'] for r in group]
            if actual!=expected:raise ValueError('ActualGPU encoder output mask lengths differ from metadata')
            if not torch.isfinite(out.last_hidden_state).all():raise ValueError('Nonfinite actualencoder outputs')
            for row,mel,t in zip(group,input_lengths,actual):checks.append({'source_id':row['source_id'],'actual_mel_frames':mel,'actual_encoder_frames':t,'duration':row['duration']})
            del out,inputs,audio;torch.cuda.synchronize()
            if torch.cuda.memory_reserved()>4e9:torch.cuda.empty_cache()
            telemetry=sample_memory(os.getpid(),'bounded_TRAIN_metadata_GPU_mask_verification')
            reason,observation=guard.observe(telemetry['system_memory'],telemetry['worker_private_memory'],observed_monotonic=telemetry['observed_monotonic'])
            reason=reason or worker_process_commit_reason(telemetry['calling_process_commit_memory'],os.getpid())
            if reason:raise RuntimeError('Resource guard:'+reason)
            report.update(last_owned_memory=telemetry,resource_observation=observation)
        if any(sha(D/name)!=value for name,value in sources.items()):raise ValueError('Source changed during proof')
        report.update(passed=True,status='complete',rows_checked=len(checks),actual_GPU_mask_metadata_parity_passed=True,
            rows=checks,peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed',failure_type=type(error).__name__,failure_message=str(error));raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,finished_at_utc=datetime.now(timezone.utc).isoformat());write_json(args.report,report)
    print(json.dumps({'passed':True,'TRAIN_rows':16,'elapsed_seconds':report['elapsed_seconds']}))
if __name__=='__main__':main(parser().parse_args())
