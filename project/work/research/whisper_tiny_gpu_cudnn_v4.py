"""Use cuDNN for encoder attention, efficient attention for the one-token decoder."""
import json,time,os,statistics,pathlib,gc
import whisper_tiny_gpu_batch_v1 as common
from whisper_tiny_gpu_fused_v3 import FusedLanguageModel
AREA=common.AREA;ROOT=common.ROOT;write=common.write

def main():
    import torch
    from transformers import WhisperForConditionalGeneration,WhisperFeatureExtractor
    from torch.nn.attention import sdpa_kernel,SDPBackend
    guard=common.Guard();m=guard.check();assert m['commit_ratio']<.9 and m['commit_headroom_bytes']>=8_000_000_000
    guard.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    receipt=json.loads((AREA/'HF_download_receipt.json').read_text());folder=receipt['model_directory']
    model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
    ids=torch.tensor(list(model.generation_config.lang_to_id.values()),device='cuda');core=FusedLanguageModel(model,ids)
    ext=WhisperFeatureExtractor.from_pretrained(folder,local_files_only=True);clips,_=common.load_audio();waves=[]
    for _,a in clips:
        for i in range(0,len(a),64000):
            if len(a[i:i+64000])>=4000:waves.append(a[i:i+64000])
    features=ext(waves,sampling_rate=16000,return_tensors='pt').input_features.to('cuda',dtype=torch.float16)
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert not lease.get('actual_GPU_running');lease.update(actual_GPU_running=True,status='ROOT_tiny_cudnn_encoder_benchmark_v4',actual_owned_worker_pid=os.getpid());leasep.write_text(json.dumps(lease,indent=2)+'\n')
    original=torch.nn.functional.scaled_dot_product_attention
    def tuned(q,k,v,*a,**kw):
        backend=SDPBackend.CUDNN_ATTENTION if q.shape[-2]>1 else SDPBackend.EFFICIENT_ATTENTION
        with sdpa_kernel(backend):return original(q,k,v,*a,**kw)
    def measure(fn,b):
        guard.check()
        for i in range(5):fn()
        torch.cuda.synchronize();times=[]
        for _ in range(3):
            guard.check();start=time.perf_counter()
            for i in range(40):out=fn()
            torch.cuda.synchronize();times.append((time.perf_counter()-start)/40)
        return {'batch':b,'latency_s':statistics.median(times),'windows_per_second':b/statistics.median(times)}
    results=[]
    with torch.inference_mode():
        ref=torch.cat([core(features[i:i+8]) for i in range(0,len(features),8)])
        torch.nn.functional.scaled_dot_product_attention=tuned
        new=torch.cat([core(features[i:i+8]) for i in range(0,len(features),8)])
        verification={'windows':len(features),'max_probability_error_vs_fused_default':(new-ref).abs().max().item(),'top_language_agreement':(new.argmax(-1)==ref.argmax(-1)).float().mean().item(),'TR70_decision_changes':int(((new[:,list(model.generation_config.lang_to_id).index('<|tr|>')]>=.70)!=(ref[:,list(model.generation_config.lang_to_id).index('<|tr|>')]>=.70)).sum())}
        print('VERIFY '+json.dumps(verification),flush=True);assert verification['top_language_agreement']==1 and verification['max_probability_error_vs_fused_default']<.02
        for b in [1,4,8,16,32]:
            x=features[torch.arange(b,device='cuda')%len(features)].contiguous();result=measure(lambda:core(x),b);result['mode']='cudnn_encoder_efficient_decoder';results.append(result)
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(5):core(x)
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=core(x)
            def replay():g.replay();return out
            graph=measure(replay,b);graph.update(mode='cudnn_encoder_CUDA_graph',memory=guard.check());results.append(graph)
            print('RESULT '+json.dumps(results[-2:]),flush=True);write('cudnn_v4_partial.json',dict(verification=verification,results=results))
            del g,out;gc.collect();torch.cuda.empty_cache()
        torch.nn.functional.scaled_dot_product_attention=original
        write('cudnn_v4_report.json',dict(at_utc=common.now(),verification=verification,results=results,memory=guard.check()))
    guard.stop.set();del model;torch.cuda.empty_cache();lease.update(actual_GPU_running=False,status='ROOT_tiny_cudnn_encoder_benchmark_closed',actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE cudnn_v4',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:write('cudnn_v4_failure.json',dict(error=repr(e),at_utc=common.now(),pid=os.getpid()));raise
