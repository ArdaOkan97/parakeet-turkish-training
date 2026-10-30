"""Verified zero-suffix mel frontend, language-only CUDA graph, and input timing."""
import os,sys,pathlib,json,time,statistics,gc,argparse,contextlib
import whisper_tiny_gpu_batch_v1 as first
AREA=first.AREA;ROOT=first.ROOT;write=first.write;now=first.now

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batches',default='1,4,8,16,32');parser.add_argument('--out',default='optimized_v2');args=parser.parse_args()
    guard=first.Guard();m=guard.check();assert m['commit_ratio']<.90 and m['commit_headroom_bytes']>=8_000_000_000
    import numpy as np,torch
    from transformers import WhisperForConditionalGeneration,WhisperFeatureExtractor
    guard.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    receipt=json.loads((AREA/'HF_download_receipt.json').read_text());folder=receipt['model_directory']
    model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
    ext=WhisperFeatureExtractor.from_pretrained(folder,local_files_only=True)
    lang_ids=torch.tensor(list(model.generation_config.lang_to_id.values()),device='cuda');lang_weight=model.proj_out.weight.index_select(0,lang_ids).detach()
    sot=model.generation_config.decoder_start_token_id
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert lease['owner']=='/root' and not lease.get('actual_GPU_running')
    lease.update(status='ROOT_whisper_tiny_GPU_optimization_v2',actual_GPU_running=True,actual_owned_worker_pid=os.getpid());leasep.write_text(json.dumps(lease,indent=2)+'\n')
    clips,decode_s=first.load_audio();windows=[];manifest=[]
    for sid,a in clips:
        for start in range(0,len(a),64000):
            part=a[start:start+64000]
            if len(part)<4000:continue
            windows.append(part);manifest.append(dict(segment_id=sid,start_s=start/16000,duration_s=len(part)/16000))
    start=time.perf_counter();cpu_full=ext(windows,sampling_rate=16000,return_tensors='pt').input_features;CPU_mel_s=time.perf_counter()-start
    raw=np.zeros((len(windows),64400),np.float32)
    for i,w in enumerate(windows):raw[i,:len(w)]=w
    pinned=torch.from_numpy(raw).pin_memory();reference=cpu_full.to('cuda',dtype=torch.float16);del cpu_full
    mel_filters=torch.tensor(ext.mel_filters,device='cuda',dtype=torch.float32);hann=torch.hann_window(400,device='cuda')
    def mel(waves):
        # After every 4s window, >=400 zero samples cover the final STFT support.
        # All remaining 30s padding frames have exactly the same normalized log floor.
        spec=torch.stft(waves,400,160,window=hann,return_complex=True)
        power=(spec[...,:-1].abs()**2).contiguous();log=(mel_filters.T@power).clamp_min(1e-10).log10()
        floor=log.amax(dim=(-2,-1),keepdim=True)-8.;log=torch.maximum(log,floor)
        log=(log+4.)/4.;padding=((floor+4.)/4.).expand(log.shape[0],80,3000-log.shape[-1])
        return torch.cat((log,padding),dim=-1).to(torch.float16)
    def infer(x):
        decoder=torch.full((x.shape[0],1),sot,dtype=torch.long,device='cuda')
        h=model.model(input_features=x,decoder_input_ids=decoder,use_cache=False).last_hidden_state[:,0]
        return torch.nn.functional.linear(h,lang_weight).float().softmax(-1)
    def bench(fn,b,target=.7):
        for _ in range(3):fn()
        torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize();single=time.perf_counter()-start
        n=max(3,min(100,int(target/max(single,.0001))));values=[]
        for _ in range(3):
            guard.check();start=time.perf_counter()
            for i in range(n):out=fn()
            torch.cuda.synchronize();values.append((time.perf_counter()-start)/n)
        return dict(batch=b,latency_s=statistics.median(values),windows_per_second=b/statistics.median(values),repeats=n)
    results=[]
    with torch.inference_mode():
        cpu_probs=torch.cat([infer(reference[i:i+8]) for i in range(0,len(windows),8)])
        converted=[];max_mel_error=0
        for i in range(0,len(windows),8):
            f=mel(pinned[i:i+8].to('cuda',non_blocking=True));max_mel_error=max(max_mel_error,(f-reference[i:i+8]).abs().max().item());converted.append(f)
        optimized_features=torch.cat(converted);new_probs=torch.cat([infer(optimized_features[i:i+8]) for i in range(0,len(windows),8)])
        verification=dict(windows=len(windows),max_mel_error_after_FP16=max_mel_error,max_probability_error=(cpu_probs-new_probs).abs().max().item(),top_language_agreement=(cpu_probs.argmax(-1)==new_probs.argmax(-1)).float().mean().item(),full3000_encoder_frames_preserved=True)
        assert verification['top_language_agreement']==1 and max_mel_error<=.0025,verification
        write(args.out+'_frontend_verification.json',verification);print('FRONTEND '+json.dumps(verification),flush=True)
        for b in map(int,args.batches.split(',')):
            guard.check();idx=torch.arange(b)%len(windows);x=optimized_features[idx.to('cuda')].contiguous();pcm=pinned[idx].contiguous().pin_memory();device_pcm=pcm.to('cuda',non_blocking=True)
            eager=bench(lambda:infer(x),b);eager['mode']='eager_precomputed_mel';results.append(eager)
            frontend=bench(lambda:mel(device_pcm),b);frontend['mode']='GPU_mel_only';results.append(frontend)
            integrated=bench(lambda:infer(mel(pcm.to('cuda',non_blocking=True))),b);integrated['mode']='CPU_to_GPU_plus_mel_plus_inference';results.append(integrated)
            static=x.clone();side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(4):infer(static)
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):graph_output=infer(static)
                def replay():
                    static.copy_(x);graph.replay();return graph_output
                replay();torch.cuda.synchronize();correct=infer(x);diff=(graph_output-correct).abs().max().item();assert torch.equal(graph_output.argmax(-1),correct.argmax(-1))
                graph_result=bench(replay,b);graph_result.update(mode='CUDA_graph_with_input_copy',max_probability_error_vs_eager=diff,memory=guard.check());results.append(graph_result)
            finally:
                del graph;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
            write(args.out+'_partial.json',dict(results=results,verification=verification))
            print('BATCH '+json.dumps([r for r in results if r['batch']==b]),flush=True)
        # Real cached original-Opus decode + feature extraction + prediction, no feature reuse.
        complete=[]
        for b in [4,8,16]:
            timings=[]
            for trial in range(3):
                guard.check();start=time.perf_counter();actual_clips,decode=first.load_audio();parts=[]
                for sid,a in actual_clips:
                    for i in range(0,len(a),64000):
                        if len(a[i:i+64000])>=4000:parts.append(a[i:i+64000])
                for i in range(0,len(parts),b):
                    chunk=parts[i:i+b];buf=torch.zeros((len(chunk),64400),pin_memory=True)
                    for j,w in enumerate(chunk):buf[j,:len(w)].copy_(torch.from_numpy(w))
                    out=infer(mel(buf.to('cuda',non_blocking=True))).cpu()
                torch.cuda.synchronize();timings.append(time.perf_counter()-start)
            complete.append(dict(batch=b,total_s=statistics.median(timings),clips=len(clips),windows=len(parts),clips_per_s=len(clips)/statistics.median(timings),audio_seconds_per_wall_second=sum(len(a)/16000 for _,a in actual_clips)/statistics.median(timings),scope='44 cached original-Opus packets incl actual decode, collation, GPU mel, language inference and probability CPU copy; excludes original archive seeks, VAD and durable ledger writes'))
            print('END_TO_END '+json.dumps(complete[-1]),flush=True)
        report=dict(kind='optimized_GPU_tiny_benchmark_v2',at_utc=now(),unique_clips=len(clips),unique_windows=len(windows),verification=verification,results=results,cached_Opus_end_to_end=complete,baseline_CPU_mel_s=CPU_mel_s,baseline_cached_decode_s=decode_s,memory=guard.check(),model_revision=receipt['revision'],full_encoder_length_preserved=True,quantized=False,speech_VAD_applied=False)
        write(args.out+'_report.json',report)
    guard.stop.set();del model;torch.cuda.empty_cache();lease.update(actual_GPU_running=False,status='ROOT_whisper_tiny_GPU_optimization_window_closed',actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE '+str(AREA/(args.out+'_report.json')),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write('v2_failure.json',dict(error=repr(e),at_utc=now(),pid=os.getpid()));raise
